use candle_core::{shape::Dim, DType, Device, Error, IndexOp, Result, Tensor, D};
use std::fmt::Display;

/// Reduction Type
#[derive(Debug)]
pub enum Reduction {
    None,
    Sum,
    Mean,
    TokenMean,
}

impl Default for Reduction {
    fn default() -> Self {
        Reduction::Sum
    }
}

// -----------------------------------------------------------------------------

/// CRF
/// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/torchcrf/__init__.py#L9
pub struct CRF {
    pub(crate) num_tags: usize,
    pub(crate) batch_first: bool,

    pub(crate) start_transitions: Tensor,
    pub(crate) end_transitions: Tensor,
    pub(crate) transitions: Tensor,
}

/// CRF
/// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/torchcrf/__init__.py#L60
impl Display for CRF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CRF(num_tags: {}, batch_first: {})",
            self.num_tags, self.batch_first
        )
    }
}

impl CRF {
    /// Create a new CRF
    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/torchcrf/__init__.py#L38
    pub fn new(num_tags: usize, batch_first: bool, device: &Device) -> Result<Self> {
        if num_tags == 0 {
            return Err(Error::Msg("num_tags must be greater than 0".to_string()));
        }
        let start_transitions =
            Tensor::zeros(num_tags, DType::F64, &device)?.rand_like(-0.1, 1.0)?;
        let end_transitions = Tensor::zeros(num_tags, DType::F64, &device)?.rand_like(-0.1, 1.0)?;
        let transitions =
            Tensor::zeros((num_tags, num_tags), DType::F64, &device)?.rand_like(-0.1, 1.0)?;

        Ok(Self {
            num_tags,
            batch_first,
            start_transitions,
            end_transitions,
            transitions,
        })
    }

    /// validate
    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/torchcrf/__init__.py#L142
    fn validate(
        &self,
        emissions: &Tensor,
        tags: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<()> {
        let (d1, d2, d3) = emissions.dims3()?; // check if the tensor has 3 dimensions

        if d3 != self.num_tags {
            return Err(Error::Msg(format!(
                "expected last dimension of emissions is {}, got {}",
                self.num_tags, d3
            ))); // check if the last dimension of the tensor is equal to the number of tags
        }

        if let Some(tags) = tags {
            if tags.dtype() != DType::I64 {
                return Err(Error::Msg("tags must be of type i64".to_string()));
            }

            let (tag_d1, tag_d2) = tags.dims2()?; // check if the tensor has 2 dimensions
            if (d1, d2) != (tag_d1, tag_d2) {
                return Err(Error::Msg(format!(
                    "the first two dimensions of emissions and tags must match, got ({}, {}) and ({}, {})",
                    d1, d2, d1, d2
                )));
            }
        }

        if let Some(mask) = mask {
            if mask.dtype() != DType::U8 {
                return Err(Error::Msg("mask must be of type u8".to_string()));
            }
            let (mask_d1, mask_d2) = mask.dims2()?; // check if the tensor has 2 dimensions
            if (d1, d2) != (mask_d1, mask_d2) {
                return Err(Error::Msg(format!(
                    "the first two dimensions of emissions and mask must match, got ({}, {}) and ({}, {})",
                    d1, d2, mask_d1, mask_d2
                )));
            }

            let no_empty_seq = !self.batch_first && all(&mask.i(0)?)?;
            let no_empty_seq_bf = self.batch_first && all(&mask.i((.., 0))?)?;

            if !no_empty_seq && !no_empty_seq_bf {
                return Err(Error::Msg(
                    "mask of the first timestep must all be on".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// compute_score
    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/torchcrf/__init__.py#L172
    fn compute_score(&self, emissions: &Tensor, tags: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (d1, d2, d3) = emissions.dims3()?;
        let (seq_length, batch_size) = tags.dims2()?;
        assert_eq!(d1, seq_length);
        assert_eq!(d2, batch_size);
        assert_eq!(d3, self.num_tags);
        assert_eq!(mask.shape(), tags.shape());
        assert!(all(&mask.i(0)?)?);

        let mask = mask.to_dtype(emissions.dtype())?;

        let mut score = self.start_transitions.i(&tags.i(0)?)?;

        let z = multi_index(&emissions.i((0, 0..batch_size))?, &tags.i(0)?)?;

        score = score.broadcast_add(&z)?;

        for i in 1..seq_length {
            let z = multi_index(&self.transitions.i(&tags.i(i - 1)?)?, &tags.i(i)?)?;
            score = score.broadcast_add(&z.broadcast_mul(&mask.i(i)?)?)?;

            let z = multi_index(&emissions.i((i, 0..batch_size))?, &tags.i(i)?)?;
            score = score.broadcast_add(&z.broadcast_mul(&mask.i(i)?)?)?;
        }

        let seq_ends = mask
            .to_dtype(DType::I64)?
            .sum(0)?
            .broadcast_sub(&Tensor::ones(1, DType::I64, mask.device())?)?;

        let last_tags = dim2_i(
            &tags,
            (
                &seq_ends,
                &Tensor::arange(0, batch_size as i64, mask.device())?,
            ),
        )?;

        score.broadcast_add(&self.end_transitions.i(&last_tags)?)
    }

    /// compute_normalizer
    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/torchcrf/__init__.py#L211
    fn compute_normalizer(&self, emissions: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (d1, d2, d3) = emissions.dims3()?;
        let (seq_length, batch_size) = mask.dims2()?;
        assert_eq!(d1, seq_length);
        assert_eq!(d2, batch_size);
        assert_eq!(d3, self.num_tags);
        assert!(all(&mask.i(0)?)?);

        let mut score = self.start_transitions.broadcast_add(&emissions.i(0)?)?;

        for i in 1..seq_length {
            let broadcast_score = score.unsqueeze(2)?;

            let broadcast_emissions = emissions.i(i)?.unsqueeze(1)?;
            let next_score = broadcast_score
                .broadcast_add(&self.transitions)?
                .broadcast_add(&broadcast_emissions)?;

            let next_score = next_score.log_sum_exp(1)?;
            let z = mask.i(i)?.unsqueeze(1)?.broadcast_as(next_score.shape())?;
            score = z.where_cond(&next_score, &score)?;
        }

        score = score.broadcast_add(&self.end_transitions)?;
        score.log_sum_exp(1)
    }

    /// viterbi_decode
    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/torchcrf/__init__.py#L262
    fn viterbi_decode(&self, emissions: &Tensor, mask: &Tensor) -> Result<Vec<Vec<u32>>> {
        let (d1, d2, d3) = emissions.dims3()?;
        let (seq_length, batch_size) = mask.dims2()?;
        assert_eq!(d1, seq_length);
        assert_eq!(d2, batch_size);
        assert_eq!(d3, self.num_tags);
        assert!(all(&mask.i(0)?)?);

        let mut score = self.start_transitions.broadcast_add(&emissions.i(0)?)?;

        let mut history = Vec::with_capacity(seq_length);
        for i in 1..seq_length {
            let broadcast_sore = score.unsqueeze(2)?;

            let broadcast_emission = emissions.i(i)?.unsqueeze(1)?;

            let next_score = broadcast_sore
                .broadcast_add(&self.transitions)?
                .broadcast_add(&broadcast_emission)?;

            let (next_score, indices) = max_indices(&next_score, 1)?;

            let z = mask.i(i)?.unsqueeze(1)?.broadcast_as(next_score.shape())?;
            score = z.where_cond(&next_score, &score)?;
            history.push(indices);
        }

        score = score.broadcast_add(&self.end_transitions)?;

        let seq_ends = mask
            .to_dtype(DType::I64)?
            .sum(0)?
            .broadcast_sub(&Tensor::ones(1, DType::I64, mask.device())?)?;

        let mut best_tags_list = vec![];

        for idx in 0..batch_size {
            let best_last_tag = score.i(idx)?.argmax(0)?;

            let mut best_tags = vec![best_last_tag.to_scalar::<u32>()?];

            let z = seq_ends.i(idx)?.to_scalar::<i64>()? as usize;
            let mut a = history[..z].to_vec();
            a.reverse();
            for hist in a.iter() {
                let last_idx = *best_tags.last().unwrap() as usize;
                let best_last_tag = hist.i(idx)?.i(last_idx)?;
                best_tags.push(best_last_tag.to_scalar::<u32>()?);
            }

            best_tags.reverse();
            best_tags_list.push(best_tags);
        }

        Ok(best_tags_list)
    }

    /// decode
    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/torchcrf/__init__.py#L118
    pub fn decode(&self, emissions: &Tensor, mask: Option<&Tensor>) -> Result<Vec<Vec<u32>>> {
        self.validate(emissions, None, mask)?;
        let mask = if let Some(mask) = mask {
            mask.clone()
        } else {
            let (d1, d2, _) = emissions.dims3()?;
            Tensor::ones((d1, d2), DType::U8, emissions.device())?
        };

        let (emissions, mask) = if self.batch_first {
            (emissions.transpose(0, 1)?, mask.transpose(0, 1)?)
        } else {
            (emissions.clone(), mask.clone())
        };
        self.viterbi_decode(&emissions, &mask)
    }

    /// Forward
    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/torchcrf/__init__.py#L63
    pub fn forward(
        &self,
        emissions: &Tensor,
        tags: &Tensor,
        mask: Option<&Tensor>,
        reduction: Reduction,
    ) -> Result<Tensor> {
        self.validate(emissions, Some(tags), mask)?;
        let mask = if let Some(mask) = mask {
            mask.clone()
        } else {
            Tensor::ones_like(tags)?.to_dtype(DType::U8)?
        };

        let (emissions, tags, mask) = if self.batch_first {
            (
                emissions.transpose(0, 1)?,
                tags.transpose(0, 1)?,
                mask.transpose(0, 1)?,
            )
        } else {
            (emissions.clone(), tags.clone(), mask.clone())
        };

        let numerator = self.compute_score(&emissions, &tags, &mask)?;
        let denominator = self.compute_normalizer(&emissions, &mask)?;

        let llh = numerator.broadcast_sub(&denominator)?;

        match reduction {
            Reduction::Sum => llh.sum_all(),
            Reduction::Mean => llh.mean_all(),
            Reduction::TokenMean => {
                let mask = mask.to_dtype(llh.dtype())?;
                let z = mask.sum_all()?;
                llh.sum_all()?.broadcast_div(&z)
            }
            Reduction::None => Ok(llh),
        }
    }
}

// -----------------------------------------------------------------------------

pub(crate) fn all(x: &Tensor) -> Result<bool> {
    let zero = Tensor::zeros(1, x.dtype(), x.device())?;
    Ok(x.broadcast_ne(&zero)?
        .flatten_all()?
        .min(0)?
        .to_scalar::<u8>()?
        != 0)
}

// -----------------------------------------------------------------------------

pub(crate) fn multi_index(src: &Tensor, idx: &Tensor) -> Result<Tensor> {
    let index = idx.reshape((idx.dim(0)?, 1))?;
    src.gather(&index, D::Minus1)?.squeeze(D::Minus1)
}

// -----------------------------------------------------------------------------

pub(crate) fn dim2_i(src: &Tensor, idx: (&Tensor, &Tensor)) -> Result<Tensor> {
    if idx.0.dims1()? == 1 {
        let (x, y) = (
            idx.0.i(0)?.to_scalar::<i64>()? as usize,
            idx.1.i(0)?.to_scalar::<i64>()? as usize,
        );
        src.i((x, y))?.unsqueeze(D::Minus1)
    } else {
        multi_index(&src.i(idx.0)?, &idx.1)
    }
}
// -----------------------------------------------------------------------------

pub(crate) fn max_indices<D: Dim + Copy>(x: &Tensor, dim: D) -> Result<(Tensor, Tensor)> {
    let max = x.max(dim)?;
    let idx = x.argmax(dim)?;
    Ok((max, idx))
}

// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    /*
    following tests correspond to the following PyTorch script.
    https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L12

    RANDOM_SEED = 1478754

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
     */

    use super::*;
    use candle_core::{DType, Device, IndexOp, Result, Tensor};
    use itertools::Itertools;

    const EPSILON: f64 = 1e-12;

    fn assert_close(a: f64, b: f64, epsilon: f64) {
        assert!((a - b).abs() < epsilon);
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L37
    fn make_crf(
        num_tags: usize,
        batch_first: bool,
        start: Option<Tensor>,
        end: Option<Tensor>,
        transition: Option<Tensor>,
    ) -> candle_core::Result<CRF> {
        let mut crf = CRF::new(num_tags, batch_first, &candle_core::Device::Cpu)?;
        if let Some(start) = start {
            crf.start_transitions = start.to_device(&Device::Cpu)?;
        }
        if let Some(end) = end {
            crf.end_transitions = end.to_device(&Device::Cpu)?;
        }
        if let Some(transition) = transition {
            crf.transitions = transition.to_device(&Device::Cpu)?;
        }
        Ok(crf)
    }

    /// assert_all
    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L23
    fn assert_all<D: candle_core::WithDType>(x: &Tensor, lo: D, up: D) -> Result<bool> {
        assert_eq!(x.dims().len(), 1);
        let dim = x.dims1()?;
        for i in 0..dim {
            let a = x.i(i)?.to_scalar::<D>()?;
            if a < lo || a > up {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// compute_score
    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L18
    fn compute_score(crf: &CRF, emission: &Tensor, tag: &Tensor) -> candle_core::Result<Tensor> {
        assert_eq!(emission.dims().len(), 2);
        let (emission_dim1, emission_dim2) = emission.dims2()?;
        let tag_dim1 = tag.dims1()?;
        assert_eq!(emission_dim1, tag_dim1);
        assert_eq!(emission_dim2, crf.num_tags);
        assert_all(tag, 0_i64, crf.num_tags as i64 - 1)?;

        let tag_vec = tag.to_vec1::<i64>()?;

        let mut score = crf
            .start_transitions
            .i(tag_vec[0] as usize)?
            .broadcast_add(&crf.end_transitions.i(tag_vec[tag_vec.len() - 1] as usize)?)?;

        for (cur_tag, next_tag) in tag_vec.iter().zip(tag_vec.iter().skip(1)) {
            let z = crf.transitions.i((*cur_tag as usize, *next_tag as usize))?;
            score = score.broadcast_add(&z)?;
        }

        for (i, &t) in tag_vec.iter().enumerate() {
            let z = emission.i((i, t as usize))?;
            score = score.broadcast_add(&z)?;
        }

        Ok(score)
    }

    fn cartestian_product(r: Vec<i64>, repeat: usize, dev: &Device) -> Result<Vec<Tensor>> {
        use itertools::Itertools;

        if repeat <= 1 {
            return Ok(vec![Tensor::new(r.as_slice(), dev)?]);
        }

        let mut a: Vec<Vec<i64>> = r
            .iter()
            .cartesian_product(r.iter())
            .map(|(&x, &y)| vec![x, y])
            .collect();
        for _ in 2..repeat {
            a = a
                .iter()
                .cartesian_product(r.iter())
                .map(|(x, &y)| {
                    let mut z = Vec::from(x.to_owned());
                    z.push(y);
                    z
                })
                .collect();
        }
        Ok(a.iter()
            .map(|x| Tensor::new(x.as_slice(), dev).unwrap())
            .collect())
    }

    #[test]
    fn test_cartestian_product() {
        let a = (0..5_i64).collect::<Vec<_>>();
        let a = cartestian_product(a, 3, &Device::Cpu).unwrap();
        println!("{:?}", a);
    }

    /// test_minial
    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L60
    #[test]
    fn test_init_minial() {
        let num_tags = 10;
        let crf = CRF::new(num_tags, false, &Device::Cpu).unwrap();
        assert_eq!(crf.num_tags, num_tags);
        assert!(!crf.batch_first);
        assert_eq!(crf.start_transitions.dims1().unwrap(), num_tags);
        assert_eq!(crf.end_transitions.dims1().unwrap(), num_tags);
        assert_eq!(crf.transitions.dims2().unwrap(), (num_tags, num_tags));
        println!("crf:{}", crf);
    }

    /// test_full
    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L74
    #[test]
    fn test_init_full() {
        let crf = CRF::new(10, true, &Device::Cpu).unwrap();
        assert!(crf.batch_first);
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L78C9-L78C34
    #[test]
    fn test_init_nonpositive_num_tags() {
        let crf = CRF::new(0, false, &Device::Cpu);
        assert!(crf.is_err());
    }

    /// test_works_with_mask
    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L85
    #[test]
    fn test_init_works_with_mask() {
        let crf = make_crf(
            5,
            false,
            Some(Tensor::new(&[-0.0687, 0.0698, -0.0447, 0.0421, 0.0782], &Device::Cpu).unwrap()),
            Some(Tensor::new(&[0.0061, -0.0671, -0.0797, 0.0629, -0.0136], &Device::Cpu).unwrap()),
            Some(
                Tensor::new(
                    &[
                        [0.0489, -0.0002, 0.0619, 0.0458, 0.0662],
                        [0.0707, 0.0297, -0.0422, 0.0831, -0.0038],
                        [0.0439, 0.0178, -0.0754, 0.0260, 0.0681],
                        [0.0191, 0.0755, 0.0230, 0.0209, -0.0768],
                        [0.0303, 0.0592, -0.0297, 0.0681, 0.0801],
                    ],
                    &Device::Cpu,
                )
                .unwrap(),
            ),
        )
        .unwrap();

        let emissions = Tensor::new(
            &[
                [
                    [1.1699, 1.1900, -0.7254, 0.1490, -1.4910],
                    [-1.2101, 0.4538, 1.3654, 0.0135, -1.8480],
                ],
                [
                    [0.5861, -0.1651, 0.9721, 0.4464, -0.5512],
                    [-1.2701, -1.5360, 0.0037, 0.5853, -0.9926],
                ],
                [
                    [-1.7625, 0.5437, 1.6322, -1.1274, -0.1313],
                    [-0.9301, 0.8906, -2.6483, 0.5849, -1.1069],
                ],
            ],
            &Device::Cpu,
        )
        .unwrap();

        let tags = Tensor::new(&[[2_i64, 4], [3, 3], [4, 2]], &Device::Cpu).unwrap();
        let mask = Tensor::new(&[[1_u8, 1, 1], [1, 1, 0]], &Device::Cpu)
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let llh = crf
            .forward(&emissions, &tags, Some(&mask), Reduction::default())
            .unwrap();
        println!("llh: {:?}", llh);

        let emissions = emissions.transpose(0, 1).unwrap();
        let tags = tags.transpose(0, 1).unwrap();
        let mask = mask.transpose(0, 1).unwrap();

        let (a, _, _) = emissions.dims3().unwrap();
        let mut manual_llh = 0.0_f64;
        for i in 0..a {
            let emission = emissions.i(i).unwrap();
            let tag = tags.i(i).unwrap();
            let mask = mask.i(i).unwrap();

            let seq_len = mask.sum_all().unwrap().to_scalar::<u8>().unwrap() as usize;
            let emission = emission.i(..seq_len).unwrap();
            let tag = tag.i(..seq_len).unwrap();
            let numerator = compute_score(&crf, &emission, &tag)
                .unwrap()
                .to_scalar::<f64>()
                .unwrap();

            let num_tags = crf.num_tags as i64;
            let product =
                cartestian_product((0..num_tags).collect_vec(), seq_len, &Device::Cpu).unwrap();
            let all_scores = product
                .iter()
                .map(|t| compute_score(&crf, &emission, &t).unwrap());

            let denominator = all_scores
                .map(|x| x.to_scalar::<f64>().unwrap().exp())
                .sum::<f64>()
                .ln();

            manual_llh += numerator - denominator;
        }
        println!("manual_llh: {:?}", manual_llh);
        assert_close(llh.to_scalar::<f64>().unwrap(), manual_llh, EPSILON);
        llh.backward().unwrap();
    }

    /// test_works_without_mask
    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L122C9-L122C32
    #[test]
    fn test_init_work_without_mask() {
        let crf = make_crf(
            5,
            false,
            Some(Tensor::new(&[0.0266, -0.0539, 0.0572, -0.0199, -0.0167], &Device::Cpu).unwrap()),
            Some(Tensor::new(&[0.0084, 0.0892, 0.0942, -0.0179, 0.0112], &Device::Cpu).unwrap()),
            Some(
                Tensor::new(
                    &[
                        [0.0456, 0.0560, 0.0396, 0.0289, 0.0187],
                        [-0.0951, -0.0286, 0.0582, 0.0384, 0.0863],
                        [-0.0137, 0.0764, -0.0414, 0.0722, -0.0287],
                        [0.0365, -0.0033, 0.0726, -0.0620, 0.0516],
                        [0.0925, -0.0708, 0.0765, 0.0671, -0.0344],
                    ],
                    &Device::Cpu,
                )
                .unwrap(),
            ),
        )
        .unwrap();

        let emissions = Tensor::new(
            &[
                [
                    [0.5463, 2.0856, -0.6247, -1.0225, 0.5277],
                    [-0.4172, -1.4281, -0.5658, -0.5217, -0.6321],
                ],
                [
                    [0.4759, -0.8485, 1.0046, 0.0720, 0.3853],
                    [-0.7525, 0.1041, 0.2371, 0.5746, -0.5599],
                ],
                [
                    [-0.5022, -0.2030, 0.3655, 0.0714, 1.2449],
                    [0.1266, 0.6654, -1.1915, -0.1181, 0.0167],
                ],
            ],
            &Device::Cpu,
        )
        .unwrap();

        let tags = Tensor::new(&[[3_i64, 2], [3, 1], [4, 3]], &Device::Cpu).unwrap();

        let llh_no_mask = crf
            .forward(&emissions, &tags, None, Reduction::default())
            .unwrap();

        let llh_mask = crf
            .forward(
                &emissions,
                &tags,
                Some(
                    &Tensor::ones_like(&tags)
                        .unwrap()
                        .to_dtype(DType::U8)
                        .unwrap(),
                ),
                Reduction::default(),
            )
            .unwrap();

        assert_close(
            llh_no_mask.to_scalar::<f64>().unwrap(),
            llh_mask.to_scalar::<f64>().unwrap(),
            EPSILON,
        )
    }

    #[test]
    fn test_init_batched_loss() {
        let batch_size = 10;
        let crf = make_crf(
            5,
            false,
            Some(
                Tensor::new(&[-0.0695, -0.0117, -0.0021, -0.0635, -0.0328], &Device::Cpu).unwrap(),
            ),
            Some(Tensor::new(&[-0.0524, -0.0827, 0.0868, -0.0140, 0.0131], &Device::Cpu).unwrap()),
            Some(
                Tensor::new(
                    &[
                        [-0.0330, 0.0894, -0.0732, 0.0996, 0.0014],
                        [-0.0514, -0.0677, -0.0611, -0.0168, 0.0297],
                        [0.0580, -0.0224, -0.0465, -0.0527, -0.0133],
                        [0.0506, 0.0535, -0.0378, -0.0537, 0.0516],
                        [-0.0037, 0.0763, -0.0867, 0.0410, 0.0368],
                    ],
                    &Device::Cpu,
                )
                .unwrap(),
            ),
        )
        .unwrap();

        let emissions = Tensor::new(
            &[
                [
                    [
                        -1.5867e+00,
                        -4.0363e-01,
                        1.7869e-02,
                        -5.0247e-01,
                        8.2934e-01,
                    ],
                    [-8.5983e-01, -4.0548e-01, 1.3188e-01, 9.8255e-01, 2.2588e-01],
                    [
                        -6.8282e-01,
                        1.8752e+00,
                        -3.4774e-01,
                        -1.0902e+00,
                        1.7499e-01,
                    ],
                    [3.7244e-01, 1.1534e+00, 7.7696e-01, 3.4387e-01, -9.8422e-01],
                    [3.8489e-02, 8.2353e-01, -8.2190e-01, 8.6248e-02, 1.2238e-01],
                    [4.4424e-02, 1.3664e+00, -1.3658e+00, -2.4691e-01, 1.1135e+00],
                    [1.2708e+00, 2.9114e-01, 1.0744e+00, 5.3505e-02, -1.5511e-01],
                    [
                        1.3976e+00,
                        -1.1226e+00,
                        -9.2870e-01,
                        1.1908e-01,
                        -1.6336e+00,
                    ],
                    [6.0694e-01, 2.5764e-01, -6.8925e-01, 1.1807e+00, -6.5968e-01],
                    [3.5677e-01, -1.4314e+00, 9.4358e-01, 7.9112e-01, -2.1923e-01],
                ],
                [
                    [1.3654e+00, -2.3797e-01, 6.2540e-02, 1.5489e+00, -2.0502e+00],
                    [1.3639e+00, -5.9433e-01, 7.0876e-01, -4.9674e-01, 6.9055e-02],
                    [
                        2.3545e-01,
                        -4.0388e-01,
                        1.2455e+00,
                        -4.1925e-01,
                        -3.9647e-01,
                    ],
                    [
                        -6.3912e-01,
                        -1.4287e+00,
                        -2.2617e+00,
                        -5.6802e-01,
                        7.4044e-01,
                    ],
                    [3.8845e-01, -9.7110e-01, -5.7113e-01, 1.3628e+00, 7.4219e-01],
                    [
                        -7.7064e-01,
                        9.3300e-01,
                        -1.4319e+00,
                        -1.5991e+00,
                        2.6631e-01,
                    ],
                    [
                        1.7472e+00,
                        -5.9296e-01,
                        -1.3249e-03,
                        1.4543e-01,
                        -6.5364e-01,
                    ],
                    [
                        -3.5911e-01,
                        4.4189e-02,
                        -1.2928e+00,
                        -1.1482e+00,
                        1.2672e+00,
                    ],
                    [1.3452e+00, -2.3875e+00, 1.4895e+00, -7.3329e-01, 2.1750e-01],
                    [
                        -3.9819e-01,
                        4.5757e-01,
                        -5.0534e-01,
                        -3.0911e+00,
                        -1.1324e+00,
                    ],
                ],
                [
                    [
                        -3.4185e-01,
                        -1.0406e+00,
                        -4.3079e-01,
                        -4.5273e-02,
                        1.1170e+00,
                    ],
                    [
                        -8.5589e-01,
                        9.4792e-01,
                        -8.8419e-01,
                        -7.7756e-01,
                        -1.7976e-01,
                    ],
                    [-1.8891e-01, 1.7120e-01, -4.3634e-01, 1.2762e+00, 1.0334e+00],
                    [
                        2.7852e-01,
                        -1.5482e+00,
                        5.6432e-01,
                        -1.1859e+00,
                        -7.0821e-02,
                    ],
                    [3.4364e-01, 1.2222e+00, 1.0542e+00, -1.7861e-01, 6.4608e-01],
                    [-8.4590e-01, 1.4749e+00, 3.7927e-01, 2.2527e+00, -3.5637e-02],
                    [
                        4.5344e-01,
                        -1.4359e+00,
                        -2.2955e+00,
                        -9.4110e-01,
                        -8.5992e-01,
                    ],
                    [6.8505e-01, -1.5822e-01, -6.9359e-01, 5.9559e-02, 6.8955e-01],
                    [
                        -3.4006e-01,
                        1.7685e+00,
                        2.1671e-01,
                        -8.6512e-01,
                        -2.6517e-01,
                    ],
                    [1.0503e-01, 1.6486e+00, -2.4486e-01, 5.4843e-01, 1.9252e+00],
                ],
            ],
            &Device::Cpu,
        )
        .unwrap();

        let tags = Tensor::new(
            &[
                [1_i64, 0, 0, 1, 2, 1, 0, 4, 3, 0],
                [3, 2, 3, 4, 4, 1, 2, 1, 4, 1],
                [0, 1, 4, 4, 0, 4, 0, 1, 4, 1],
            ],
            &Device::Cpu,
        )
        .unwrap();

        let llh = crf
            .forward(&emissions, &tags, None, Reduction::default())
            .unwrap();

        llh.dims0().unwrap();
        println!("{:?}", llh.to_scalar::<f64>().unwrap());

        let mut total_llh = 0.0_f64;
        for i in 0..batch_size {
            let emissions = emissions.i((.., i, ..)).unwrap().unsqueeze(1).unwrap();
            println!("emissions: {:?}", emissions.to_vec3::<f64>().unwrap());
            let tags = tags.i((.., i)).unwrap().unsqueeze(1).unwrap();
            println!("tags: {:?}", tags.to_vec2::<i64>().unwrap());
            total_llh += crf
                .forward(&emissions, &tags, None, Reduction::default())
                .unwrap()
                .to_scalar::<f64>()
                .unwrap();
        }

        assert_close(llh.to_scalar::<f64>().unwrap(), total_llh, EPSILON);
    }

    #[test]
    fn test_arrage() {
        let a = Tensor::arange(0_i64, 1, &Device::Cpu).unwrap();
        println!("{:?}", a.to_vec1::<i64>().unwrap());
    }
}
