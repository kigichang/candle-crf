use candle_core::{shape::Dim, DType, Device, Error, IndexOp, Result, Tensor, D};
use candle_nn::{Init, VarBuilder};
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
        Self::new_with_dtype(num_tags, batch_first, DType::F32, device)
    }

    pub fn new_with_dtype(
        num_tags: usize,
        batch_first: bool,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        {
            use DType::*;
            match dtype {
                #[cfg(any(feature = "cuda", feature = "metal"))]
                F32 | F64 => {}
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                BF16 | F16 | F32 | F64 => {}
                _ => return Err(Error::UnsupportedDTypeForOp(dtype, "unsupported dtype")),
            }
        }

        if num_tags == 0 {
            return Err(Error::Msg("num_tags must be greater than 0".to_string()));
        }
        let start_transitions = Tensor::zeros(num_tags, dtype, &device)?.rand_like(-0.1, 1.0)?;
        let end_transitions = Tensor::zeros(num_tags, dtype, &device)?.rand_like(-0.1, 1.0)?;
        let transitions =
            Tensor::zeros((num_tags, num_tags), dtype, &device)?.rand_like(-0.1, 1.0)?;

        Ok(Self {
            num_tags,
            batch_first,
            start_transitions,
            end_transitions,
            transitions,
        })
    }

    pub fn load(num_tags: usize, batch_first: bool, vb: VarBuilder) -> Result<Self> {
        let start_transitions = vb.get_with_hints(
            num_tags,
            "start_transitions",
            Init::Uniform {
                lo: -0.1_f64,
                up: 1.0_f64,
            },
        )?;

        let end_transitions = vb.get_with_hints(
            num_tags,
            "end_transitions",
            Init::Uniform {
                lo: -0.1_f64,
                up: 1.0_f64,
            },
        )?;

        let transitions = vb.get_with_hints(
            (num_tags, num_tags),
            "transitions",
            Init::Uniform {
                lo: -0.1_f64,
                up: 1.0_f64,
            },
        )?;

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
        {
            let dtype_transitions = self.transitions.dtype();
            let dtype_emissions = emissions.dtype();
            if dtype_transitions != dtype_emissions {
                return Err(Error::Msg(format!(
                    "emissions and CRF must have the same dtype, expected {:?}, got {:?}",
                    dtype_transitions, dtype_emissions
                )));
            }
        }

        {
            // check if the tensor has 3 dimensions
            let dims = emissions.dims().len();
            if dims != 3 {
                return Err(Error::Msg(format!(
                    "emissions must have 3 dimensions, got {}",
                    dims
                )));
            }
        }

        let (d1, d2, d3) = emissions.dims3()?;

        if d3 != self.num_tags {
            // check if the last dimension of the tensor is equal to the number of tags
            return Err(Error::Msg(format!(
                "expected last dimension of emissions is {}, got {}",
                self.num_tags, d3
            )));
        }

        if let Some(tags) = tags {
            if tags.dtype() != DType::I64 {
                return Err(Error::Msg("tags must be of type i64".to_string()));
            }

            if tags.dims().len() != 2 {
                // check if the tensor has 2 dimensions
                return Err(Error::Msg(format!(
                    "tags must have 2 dimensions, got {}",
                    tags.dims().len()
                )));
            }

            let (tag_d1, tag_d2) = tags.dims2()?;
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

            if mask.dims().len() != 2 {
                // check if the tensor has 2 dimensions
                return Err(Error::Msg(format!(
                    "mask must have 2 dimensions, got {}",
                    mask.dims().len()
                )));
            }

            let (mask_d1, mask_d2) = mask.dims2()?;
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

        let z = gather(&emissions.i((0, 0..batch_size))?, &tags.i(0)?)?;

        score = score.broadcast_add(&z)?;

        for i in 1..seq_length {
            let z = gather(&self.transitions.i(&tags.i(i - 1)?)?, &tags.i(i)?)?;
            score = score.broadcast_add(&z.broadcast_mul(&mask.i(i)?)?)?;

            let z = gather(&emissions.i((i, 0..batch_size))?, &tags.i(i)?)?;
            score = score.broadcast_add(&z.broadcast_mul(&mask.i(i)?)?)?;
        }

        let seq_ends = mask
            .to_dtype(DType::I64)?
            .sum(0)?
            .broadcast_sub(&Tensor::ones(1, DType::I64, mask.device())?)?;

        let last_tags = gather(
            &tags.i(&seq_ends)?,
            &Tensor::arange(0, batch_size as i64, mask.device())?,
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
    let zero = x.zeros_like()?;
    Ok(x.broadcast_ne(&zero)?
        .flatten_all()?
        .min(0)?
        .to_scalar::<u8>()?
        != 0)
}

// -----------------------------------------------------------------------------

pub(crate) fn gather(src: &Tensor, idx: &Tensor) -> Result<Tensor> {
    let index = idx.reshape((idx.dim(0)?, 1))?;
    src.gather(&index, D::Minus1)?.squeeze(D::Minus1)
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
    use anyhow::Result;
    use candle_core::{utils, DType, Device, IndexOp, Tensor};
    use itertools::Itertools;

    #[cfg(any(feature = "cuda", feature = "metal"))]
    const OK_TYPES: [DType; 2] = [DType::F32, DType::F64];
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    const OK_TYPES: [DType; 4] = [DType::F32, DType::F64, DType::F16, DType::BF16];

    #[cfg(any(feature = "cuda", feature = "metal"))]
    const FAIL_TYPES: [DType; 5] = [DType::U8, DType::U32, DType::I64, DType::F16, DType::BF16];
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    const FAIL_TYPES: [DType; 3] = [DType::U8, DType::U32, DType::I64];

    enum DTypeCase {
        DType(DType),
        PyTorchCRF,
    }

    fn epsilon(type_case: DTypeCase) -> f64 {
        match type_case {
            DTypeCase::DType(dtype) => match dtype {
                DType::F64 => 1e-6,
                DType::F32 => 1e-4,
                DType::F16 => 1e-2,
                DType::BF16 => 1e-1,
                _ => panic!("dtype not supported"),
            },
            DTypeCase::PyTorchCRF => 1e-3,
        }
    }

    fn assert_tensor_close(a: &Tensor, b: &Tensor, epsilon: f64) -> Result<()> {
        assert!(a.dtype() == b.dtype());
        assert_eq!(a.shape(), b.shape());
        let epsilon = Tensor::full(epsilon, a.shape(), a.device())?.to_dtype(a.dtype())?;
        let diff = a.broadcast_sub(b)?.abs()?;
        let result = all(&diff.broadcast_le(&epsilon)?)?;
        assert!(result);
        Ok(())
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

    fn cat_scalar_tensor(tensors: Vec<Tensor>) -> candle_core::Result<Tensor> {
        let tensors: Vec<Tensor> = tensors
            .into_iter()
            .map(|t| t.unsqueeze(0).unwrap())
            .collect();
        Tensor::cat(&tensors, 0)
    }

    fn use_gpu(gpu: bool) -> candle_core::Result<Device> {
        if gpu {
            if utils::cuda_is_available() {
                println!("CUDA is available");
                Device::new_cuda(0)
            } else if utils::metal_is_available() {
                println!("Metal is available");
                Device::new_metal(0)
            } else {
                println!("CUDA and Metal are not available, using CPU");
                Ok(Device::Cpu)
            }
        } else {
            println!("Using CPU");
            Ok(Device::Cpu)
        }
    }

    #[test]
    fn test_cat_scalar_tensor() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        let mut lst = vec![];
        for i in 0..10 {
            let x = Tensor::full(i as f32, (), &device)?;
            lst.push(x);
        }

        let result = cat_scalar_tensor(lst)?;
        assert_eq!(
            result.to_vec1::<f32>().unwrap(),
            vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
        );
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L37
    fn make_crf(
        num_tags: usize,
        batch_first: bool,
        start: Option<Tensor>,
        end: Option<Tensor>,
        transition: Option<Tensor>,
        device: &Device,
    ) -> candle_core::Result<CRF> {
        let mut crf = CRF::new(num_tags, batch_first, device)?;
        if let Some(start) = start {
            crf.start_transitions = start
        }
        if let Some(end) = end {
            crf.end_transitions = end
        }
        if let Some(transition) = transition {
            crf.transitions = transition
        }
        Ok(crf)
    }

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

    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L18
    fn compute_score(crf: &CRF, emission: &Tensor, tag: &Tensor) -> Result<Tensor> {
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

    #[test]
    fn test_init_with_dtype() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        for dtype in OK_TYPES {
            assert!(CRF::new_with_dtype(10, false, dtype, &device).is_ok());
        }

        for dtype in FAIL_TYPES {
            assert!(CRF::new_with_dtype(10, false, dtype, &device).is_err());
        }
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L60
    #[test]
    fn test_init_minial() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        let num_tags = 10;
        let crf = CRF::new(num_tags, false, &device)?;
        assert_eq!(crf.num_tags, num_tags);
        assert!(!crf.batch_first);
        assert_eq!(crf.start_transitions.dims1()?, num_tags);
        assert_eq!(crf.end_transitions.dims1()?, num_tags);
        assert_eq!(crf.transitions.dims2()?, (num_tags, num_tags));
        println!("crf:{}", crf);
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L74
    #[test]
    fn test_init_full() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        let crf = CRF::new(10, true, &device)?;
        assert!(crf.batch_first);
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L78C9-L78C34
    #[test]
    fn test_init_nonpositive_num_tags() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        let crf = CRF::new(0, false, &device);
        assert!(crf.is_err());

        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L85
    fn forward_works_with_mask(dtype: DType, device: &Device) -> Result<()> {
        let crf = make_crf(
            5,
            false,
            Some(
                Tensor::new(&[-0.0687, 0.0698, -0.0447, 0.0421, 0.0782], device)?
                    .to_dtype(dtype)?,
            ),
            Some(
                Tensor::new(&[0.0061, -0.0671, -0.0797, 0.0629, -0.0136], device)?
                    .to_dtype(dtype)?,
            ),
            Some(
                Tensor::new(
                    &[
                        [0.0489, -0.0002, 0.0619, 0.0458, 0.0662],
                        [0.0707, 0.0297, -0.0422, 0.0831, -0.0038],
                        [0.0439, 0.0178, -0.0754, 0.0260, 0.0681],
                        [0.0191, 0.0755, 0.0230, 0.0209, -0.0768],
                        [0.0303, 0.0592, -0.0297, 0.0681, 0.0801],
                    ],
                    device,
                )?
                .to_dtype(dtype)?,
            ),
            device,
        )?;

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
            device,
        )?
        .to_dtype(dtype)?;

        let tags = Tensor::new(&[[2_i64, 4], [3, 3], [4, 2]], device)?;
        let mask = Tensor::new(&[[1_u8, 1, 1], [1, 1, 0]], device)?.transpose(0, 1)?;
        let llh = crf.forward(&emissions, &tags, Some(&mask), Reduction::default())?;
        println!("llh: {:?}", llh);

        let emissions = emissions.transpose(0, 1)?;
        let tags = tags.transpose(0, 1)?;
        let mask = mask.transpose(0, 1)?;

        let (a, _, _) = emissions.dims3()?;
        let mut manual_llh = Tensor::zeros(llh.shape(), llh.dtype(), llh.device())?;

        for i in 0..a {
            let emission = emissions.i(i)?;
            let tag = tags.i(i)?;
            let mask = mask.i(i)?;

            let seq_len = mask.sum_all().unwrap().to_scalar::<u8>()? as usize;
            let emission = emission.i(..seq_len)?;
            let tag = tag.i(..seq_len)?;
            let numerator = compute_score(&crf, &emission, &tag)?;

            let num_tags = crf.num_tags as i64;
            let product = cartestian_product((0..num_tags).collect_vec(), seq_len, device)?;
            let all_scores = product
                .iter()
                .map(|t| compute_score(&crf, &emission, &t).unwrap());

            let mut denominator =
                Tensor::zeros(numerator.shape(), numerator.dtype(), numerator.device())?;

            for s in all_scores.into_iter() {
                denominator = denominator.broadcast_add(&s.exp()?)?;
            }
            let denominator = denominator.log()?;

            manual_llh = manual_llh.broadcast_add(&numerator.broadcast_sub(&denominator)?)?;
        }
        println!("manual_llh: {:?}", manual_llh);
        assert_tensor_close(&llh, &manual_llh, epsilon(DTypeCase::DType(llh.dtype())))?;

        if llh.dtype() == DType::F32 {
            let manual_llh = Tensor::full(-11.0540_f32, llh.shape(), llh.device())?;
            println!("Compare with pytorch-crf: {:?}, {:?}", llh, manual_llh);
            assert_tensor_close(&llh, &manual_llh, epsilon(DTypeCase::PyTorchCRF))?;
        }
        llh.backward()?;
        Ok(())
    }

    #[test]
    fn test_forward_works_with_mask() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        for dtype in OK_TYPES {
            forward_works_with_mask(dtype, &device)?;
        }
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L122C9-L122C32
    fn forward_works_without_mask(dtype: DType, device: &Device) -> Result<()> {
        let crf = make_crf(
            5,
            false,
            Some(
                Tensor::new(&[0.0266, -0.0539, 0.0572, -0.0199, -0.0167], device)?
                    .to_dtype(dtype)?,
            ),
            Some(Tensor::new(&[0.0084, 0.0892, 0.0942, -0.0179, 0.0112], device)?.to_dtype(dtype)?),
            Some(
                Tensor::new(
                    &[
                        [0.0456, 0.0560, 0.0396, 0.0289, 0.0187],
                        [-0.0951, -0.0286, 0.0582, 0.0384, 0.0863],
                        [-0.0137, 0.0764, -0.0414, 0.0722, -0.0287],
                        [0.0365, -0.0033, 0.0726, -0.0620, 0.0516],
                        [0.0925, -0.0708, 0.0765, 0.0671, -0.0344],
                    ],
                    device,
                )?
                .to_dtype(dtype)?,
            ),
            device,
        )?;

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
            device,
        )?
        .to_dtype(dtype)?;

        let tags = Tensor::new(&[[3_i64, 2], [3, 1], [4, 3]], device)?;

        let llh_no_mask = crf.forward(&emissions, &tags, None, Reduction::default())?;

        let llh_mask = crf.forward(
            &emissions,
            &tags,
            Some(&Tensor::ones_like(&tags)?.to_dtype(DType::U8)?),
            Reduction::default(),
        )?;

        println!("llh_no_mask: {:?}", llh_no_mask);
        println!("llh_mask: {:?}", llh_mask);

        if llh_no_mask.dtype() == DType::F32 {
            let manual_llh = Tensor::full(-11.0571_f32, llh_no_mask.shape(), llh_no_mask.device())?;
            println!(
                "compare with pytorch-crf: {:?}, {:?}",
                llh_no_mask, manual_llh
            );
            assert_tensor_close(&llh_no_mask, &manual_llh, epsilon(DTypeCase::PyTorchCRF))?;
        }

        assert_tensor_close(
            &llh_no_mask,
            &llh_mask,
            epsilon(DTypeCase::DType(llh_no_mask.dtype())),
        )
    }

    #[test]
    fn test_forward_works_without_mask() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        for dtype in OK_TYPES {
            forward_works_without_mask(dtype, &device)?;
        }
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L135
    fn forward_batched_loss(dtype: DType, device: &Device) -> Result<()> {
        let batch_size = 10;
        let crf = make_crf(
            5,
            false,
            Some(
                Tensor::new(&[-0.0695, -0.0117, -0.0021, -0.0635, -0.0328], device)?
                    .to_dtype(dtype)?,
            ),
            Some(
                Tensor::new(&[-0.0524, -0.0827, 0.0868, -0.0140, 0.0131], device)?
                    .to_dtype(dtype)?,
            ),
            Some(
                Tensor::new(
                    &[
                        [-0.0330, 0.0894, -0.0732, 0.0996, 0.0014],
                        [-0.0514, -0.0677, -0.0611, -0.0168, 0.0297],
                        [0.0580, -0.0224, -0.0465, -0.0527, -0.0133],
                        [0.0506, 0.0535, -0.0378, -0.0537, 0.0516],
                        [-0.0037, 0.0763, -0.0867, 0.0410, 0.0368],
                    ],
                    device,
                )?
                .to_dtype(dtype)?,
            ),
            device,
        )?;

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
            device,
        )?
        .to_dtype(dtype)?;

        let tags = Tensor::new(
            &[
                [1_i64, 0, 0, 1, 2, 1, 0, 4, 3, 0],
                [3, 2, 3, 4, 4, 1, 2, 1, 4, 1],
                [0, 1, 4, 4, 0, 4, 0, 1, 4, 1],
            ],
            device,
        )?;

        let llh = crf.forward(&emissions, &tags, None, Reduction::default())?;

        llh.dims0()?;
        let mut total_llh = Tensor::zeros(llh.shape(), llh.dtype(), llh.device())?;
        for i in 0..batch_size {
            let emissions = emissions
                .i((.., i, ..))?
                .contiguous()? // force contiguous to fix tensor indexer.
                .unsqueeze(1)?;

            let tags = tags
                .i((.., i))?
                .contiguous()? // force contiguous to fix tensor indexer.
                .unsqueeze(1)?;

            total_llh = total_llh.broadcast_add(&crf.forward(
                &emissions,
                &tags,
                None,
                Reduction::default(),
            )?)?;
        }

        println!("llh: {:?}", llh);
        println!("total_llh: {:?}", total_llh);

        if llh.dtype() == DType::F32 {
            let manual_llh = Tensor::full(-49.2024_f32, llh.shape(), llh.device())?;
            println!("compare with pytorch-crf: {:?}, {:?}", llh, manual_llh);
            assert_tensor_close(&llh, &manual_llh, epsilon(DTypeCase::PyTorchCRF))?;
        }

        assert_tensor_close(&llh, &total_llh, epsilon(DTypeCase::DType(llh.dtype())))
    }

    #[test]
    fn test_forward_batched_loss() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        for dtype in OK_TYPES {
            forward_batched_loss(dtype, &device)?;
        }
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L159
    fn forward_reduction_none(dtype: DType, device: &Device) -> Result<()> {
        let crf = make_crf(
            5,
            false,
            Some(
                Tensor::new(&[0.0432, 0.0507, -0.0286, 0.0476, -0.0603], device)?
                    .to_dtype(dtype)?,
            ),
            Some(
                Tensor::new(&[0.0824, 0.0845, -0.0180, -0.0773, 0.0414], device)?
                    .to_dtype(dtype)?,
            ),
            Some(
                Tensor::new(
                    &[
                        [-0.0894, 0.0512, 0.0066, 0.0534, -0.0182],
                        [0.0043, 0.0328, -0.0805, -0.0945, 0.0495],
                        [-0.0020, 0.0416, -0.0441, 0.0390, 0.0690],
                        [-0.0260, 0.0720, 0.0017, -0.0552, -0.0470],
                        [0.0104, 0.0299, -0.0182, -0.0515, 0.0424],
                    ],
                    device,
                )?
                .to_dtype(dtype)?,
            ),
            device,
        )?;

        let emissions = Tensor::new(
            &[
                [
                    [-0.2560, 1.2459, -2.0063, -0.5449, 1.0978],
                    [0.7233, -0.6967, 0.3394, 0.7784, -3.0362],
                ],
                [
                    [-1.3406, -1.1565, 0.0870, 1.8249, 1.3740],
                    [-1.3396, -1.0208, 0.6608, -0.5917, 1.3850],
                ],
                [
                    [1.1436, 0.4477, 0.6606, 1.5938, -0.1054],
                    [-0.5401, 1.1908, -1.7266, -0.5858, -1.4395],
                ],
            ],
            device,
        )?
        .to_dtype(dtype)?;

        let tags = Tensor::new(&[[1_i64, 3], [1, 3], [1, 2]], device)?;
        let llh = crf.forward(&emissions, &tags, None, Reduction::None)?;
        println!("llh: {:?}", llh);
        let (seq_length, batch_size) = tags.dims2()?;
        assert_eq!(llh.dims1()?, batch_size);

        let emissions = emissions.transpose(0, 1)?;
        let tags = tags.transpose(0, 1)?;

        let (a, _, _) = emissions.dims3()?;
        let mut manual_llh = vec![];
        for i in 0..a {
            let emission = emissions.i(i)?;
            let tag = tags.i(i)?;

            let numerator = compute_score(&crf, &emission, &tag)?;

            let num_tags = crf.num_tags as i64;
            let product = cartestian_product((0..num_tags).collect_vec(), seq_length, device)?;

            let all_scores = product
                .iter()
                .map(|t| compute_score(&crf, &emission, &t).unwrap());

            let mut denominator = numerator.zeros_like()?;

            for s in all_scores.into_iter() {
                denominator = denominator.broadcast_add(&s.exp()?)?;
            }

            let denominator = denominator.log()?;
            manual_llh.push((numerator - denominator)?);
        }

        let manual_llh = cat_scalar_tensor(manual_llh)?;
        println!("manual_llh: {:?}", manual_llh);
        if dtype == DType::F32 {
            let manual_llh = Tensor::new(&[-6.3064_f32, -7.0368], device)?;
            println!("compare with pytorch-crf: {:?}, {:?}", llh, manual_llh);
            assert_tensor_close(&llh, &manual_llh, epsilon(DTypeCase::PyTorchCRF))?;
        }
        assert_tensor_close(&llh, &manual_llh, epsilon(DTypeCase::DType(llh.dtype())))
    }

    #[test]
    fn test_forward_reduction_none() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        for dtype in OK_TYPES {
            forward_reduction_none(dtype, &device)?;
        }
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L192
    fn forward_reduction_mean(dtype: DType, device: &Device) -> Result<()> {
        let crf = make_crf(
            5,
            false,
            Some(
                Tensor::new(&[0.0606, -0.0597, 0.0217, -0.0760, 0.0096], device)?
                    .to_dtype(dtype)?,
            ),
            Some(
                Tensor::new(&[-0.0791, -0.0159, 0.0525, 0.0451, -0.0373], device)?
                    .to_dtype(dtype)?,
            ),
            Some(
                Tensor::new(
                    &[
                        [5.0599e-02, -1.4571e-02, 2.2383e-02, 3.3254e-02, 2.5206e-03],
                        [6.6520e-02, 7.3251e-02, 1.0225e-02, -9.4751e-02, -3.4146e-02],
                        [
                            -6.7073e-02,
                            2.9719e-02,
                            -8.5645e-02,
                            4.6357e-02,
                            -7.2483e-03,
                        ],
                        [
                            4.4980e-02,
                            -8.0436e-02,
                            6.4611e-05,
                            -5.1731e-02,
                            -8.2973e-02,
                        ],
                        [-5.0593e-02, 4.5717e-03, 6.8714e-03, 8.9858e-02, -8.2813e-02],
                    ],
                    device,
                )?
                .to_dtype(dtype)?,
            ),
            device,
        )?;

        let emissions = Tensor::new(
            &[
                [
                    [0.0535, 0.6821, -0.2587, 1.2250, 0.5327],
                    [-2.5028, 0.5942, -0.2508, 0.0597, 1.3800],
                ],
                [
                    [-0.0640, -1.3170, 0.6408, -0.1368, -0.2137],
                    [-0.3985, 0.0530, -0.0448, 0.8268, 0.7622],
                ],
                [
                    [1.4061, -0.4045, -0.3174, 0.0737, -1.8753],
                    [-1.0892, -0.8641, 0.4778, -0.4032, 0.2838],
                ],
            ],
            device,
        )?
        .to_dtype(dtype)?;

        let tags = Tensor::new(&[[3_i64, 0], [0, 3], [2, 4]], device)?;
        let llh = crf.forward(&emissions, &tags, None, Reduction::Mean)?;
        println!("llh: {:?}", llh);

        let (seq_length, batch_size) = tags.dims2().unwrap();

        let emissions = emissions.transpose(0, 1).unwrap();
        let tags = tags.transpose(0, 1).unwrap();

        let (a, _, _) = emissions.dims3().unwrap();
        let mut manual_llh = llh.zeros_like()?;
        for i in 0..a {
            let emission = emissions.i(i).unwrap();
            let tag = tags.i(i).unwrap();

            let numerator = compute_score(&crf, &emission, &tag)?;

            let num_tags = crf.num_tags as i64;

            let product =
                cartestian_product((0..num_tags).collect_vec(), seq_length, &Device::Cpu)?;

            let all_scores = product
                .iter()
                .map(|t| compute_score(&crf, &emission, &t).unwrap());

            let mut denominator = numerator.zeros_like()?;

            for s in all_scores.into_iter() {
                denominator = denominator.broadcast_add(&s.exp()?)?;
            }

            let denominator = denominator.log()?;

            manual_llh = (manual_llh + (numerator - denominator)?)?;
        }

        manual_llh = (&manual_llh
            / Tensor::full(batch_size as f64, manual_llh.shape(), manual_llh.device())?
                .to_dtype(manual_llh.dtype())?)?;

        println!("manual_llh: {:?}", manual_llh);

        if dtype == DType::F32 {
            let manual_llh = Tensor::new(-5.7756_f32, &llh.device())?;
            println!("compare with pytorch-crf: {:?}, {:?}", llh, manual_llh);
            assert_tensor_close(&llh, &manual_llh, epsilon(DTypeCase::PyTorchCRF))?;
        }
        assert_tensor_close(&llh, &manual_llh, epsilon(DTypeCase::DType(dtype)))
    }

    #[test]
    fn test_forward_reduction_mean() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        for dtype in OK_TYPES {
            forward_reduction_mean(dtype, &device)?;
        }
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L192
    fn forward_token_mean(dtype: DType, device: &Device) -> Result<()> {
        let crf = make_crf(
            5,
            false,
            Some(Tensor::new(&[0.0687, 0.0533, 0.0204, 0.0250, -0.0785], device)?.to_dtype(dtype)?),
            Some(
                Tensor::new(
                    &[4.8827e-02, -9.9134e-05, 9.3184e-02, -7.6271e-02, 3.6482e-02],
                    device,
                )?
                .to_dtype(dtype)?,
            ),
            Some(
                Tensor::new(
                    &[
                        [0.0173, -0.0058, -0.0699, -0.0374, 0.0797],
                        [-0.0405, 0.0141, -0.0002, 0.0790, 0.0205],
                        [-0.0473, 0.0554, -0.0036, 0.0878, 0.0210],
                        [0.0761, -0.0406, -0.0905, 0.0590, -0.0030],
                        [0.0613, 0.0871, -0.0343, 0.0384, 0.0485],
                    ],
                    device,
                )?
                .to_dtype(dtype)?,
            ),
            device,
        )?;

        let emissions = Tensor::new(
            &[
                [
                    [0.0110, -0.8502, 0.9678, -0.3219, -0.6029],
                    [1.0804, -1.2822, 1.4129, 0.9475, -2.6282],
                ],
                [
                    [0.8993, 0.3029, -0.0686, -0.3108, 0.6216],
                    [-2.1503, 1.4301, -0.0301, 0.3572, 0.5460],
                ],
                [
                    [1.3384, 0.8500, 0.0194, -0.6371, 0.1516],
                    [-0.7357, 0.3116, 1.5733, -0.8246, -0.4224],
                ],
            ],
            device,
        )?
        .to_dtype(dtype)?;

        let tags = Tensor::new(&[[2_i64, 2], [0, 4], [3, 3]], device)?;
        let mask = Tensor::new(&[[1_u8, 1, 1], [1, 1, 0]], device)?.transpose(0, 1)?;
        let llh = crf.forward(&emissions, &tags, Some(&mask), Reduction::TokenMean)?;
        println!("llh: {:?}", llh);

        let emissions = emissions.transpose(0, 1)?;
        let tags = tags.transpose(0, 1)?;
        let mask = mask.transpose(0, 1)?;

        let (a, _, _) = emissions.dims3()?;
        let mut manual_llh = llh.zeros_like()?;
        let mut total_tokens = 0;
        for i in 0..a {
            let emission = emissions.i(i)?;
            let tag = tags.i(i)?;
            let mask = mask.i(i)?;

            let seq_len = mask.sum_all()?.to_scalar::<u8>()? as usize;
            let emission = emission.i(..seq_len)?;
            let tag = tag.i(..seq_len)?;
            let numerator = compute_score(&crf, &emission, &tag)?;

            let num_tags = crf.num_tags as i64;
            let product = cartestian_product((0..num_tags).collect_vec(), seq_len, device)?;

            let all_scores = product
                .iter()
                .map(|t| compute_score(&crf, &emission, &t).unwrap());

            let mut denominator = numerator.zeros_like()?;
            for t in all_scores.into_iter() {
                denominator = denominator.broadcast_add(&t.exp()?)?;
            }

            let denominator = denominator.log()?;

            manual_llh = (manual_llh + (numerator - denominator)?)?;
            total_tokens += seq_len;
        }

        let total_tokens =
            Tensor::full(total_tokens as f64, manual_llh.shape(), manual_llh.device())?
                .to_dtype(manual_llh.dtype())?;

        let manual_llh = (manual_llh / total_tokens)?;
        println!("manual_llh: {:?}", manual_llh);

        if dtype == DType::F32 {
            let manual_llh = Tensor::new(-1.4603_f32, &llh.device())?;
            println!("compare with pytorch-crf: {:?}, {:?}", llh, manual_llh);
            assert_tensor_close(&llh, &manual_llh, epsilon(DTypeCase::PyTorchCRF))?;
        }
        assert_tensor_close(&llh, &manual_llh, epsilon(DTypeCase::DType(dtype)))?;
        llh.backward()?;
        Ok(())
    }

    #[test]
    fn test_forward_token_mean() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        for dtype in OK_TYPES {
            forward_token_mean(dtype, &device)?;
        }
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L263
    fn forward_batch_first(dtype: DType, device: &Device) -> Result<()> {
        let crf = make_crf(
            5,
            false,
            Some(
                Tensor::new(&[0.0384, -0.0811, -0.0291, 0.0444, 0.0943], device)?
                    .to_dtype(dtype)?,
            ),
            Some(Tensor::new(&[0.0146, 0.0455, 0.0991, 0.0640, -0.0298], device)?.to_dtype(dtype)?),
            Some(
                Tensor::new(
                    &[
                        [0.0063, 0.0014, 0.0804, -0.0385, -0.0485],
                        [0.0485, -0.0963, 0.0799, 0.0198, -0.0549],
                        [0.0016, 0.0012, -0.0411, 0.0540, -0.0823],
                        [0.0111, 0.0320, 0.0769, 0.0292, -0.0707],
                        [-0.0990, -0.0971, 0.0635, 0.0166, 0.0292],
                    ],
                    device,
                )?
                .to_dtype(dtype)?,
            ),
            device,
        )?;

        let emissions = Tensor::new(
            &[
                [
                    [-1.1338, -0.9228, 0.3260, 0.0327, -1.0345],
                    [0.1106, 0.8005, 0.3860, -0.1214, -1.8224],
                ],
                [
                    [-1.3724, -2.2578, -1.8705, -0.1109, 0.3845],
                    [-0.4223, 0.8414, -1.4423, -1.2734, 0.5193],
                ],
                [
                    [0.4189, -1.4048, -1.6877, 1.0891, 0.6978],
                    [-0.2521, -1.4185, -0.6026, 1.6335, 1.0366],
                ],
            ],
            device,
        )?
        .to_dtype(dtype)?;

        let tags = Tensor::new(&[[1_i64, 4], [1, 4], [1, 4]], device)?;
        let llh = crf.forward(&emissions, &tags, None, Reduction::default())?;
        println!("llh: {:?}", llh);

        let crf_bf = make_crf(
            5,
            true,
            Some(crf.start_transitions),
            Some(crf.end_transitions),
            Some(crf.transitions),
            device,
        )?;

        let emissions = emissions.transpose(0, 1).unwrap();
        let tags = tags.transpose(0, 1).unwrap();

        let llh_bf = crf_bf.forward(&emissions, &tags, None, Reduction::default())?;
        println!("llh_bf: {:?}", llh_bf);

        if dtype == DType::F32 {
            let llh_bf = Tensor::new(-14.8640_f32, &llh.device())?;
            println!("compare with pytorch-crf: {:?}, {:?}", llh, llh_bf);
            assert_tensor_close(&llh, &llh_bf, epsilon(DTypeCase::PyTorchCRF))?;
        }
        assert_tensor_close(&llh, &llh_bf, epsilon(DTypeCase::DType(dtype)))
    }

    #[test]
    fn test_forward_batch_first() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        for dtype in OK_TYPES {
            forward_batch_first(dtype, &device)?;
        }
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L286
    #[test]
    fn test_forward_emissions_has_bad_number_of_dimension() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        let emissions = Tensor::randn(0.0_f32, 1., (1, 2), &device)?;
        let tags = Tensor::zeros((2, 2), DType::I64, &device)?;

        let crf = make_crf(5, false, None, None, None, &device)?;
        let result = crf.forward(&emissions, &tags, None, Reduction::default());
        println!("{:?}", result);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "emissions must have 3 dimensions, got 2"
        );
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L295C9-L295C46
    #[test]
    fn test_forward_emissions_and_tags_size_mismatch() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        let emissions = Tensor::randn(0.0_f32, 1., (1, 2, 3), &device)?;
        let tags = Tensor::zeros((2, 2), DType::I64, &device)?;
        let crf = make_crf(3, false, None, None, None, &device)?;
        let result = crf.forward(&emissions, &tags, None, Reduction::default());
        println!("{:?}", result);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "the first two dimensions of emissions and tags must match, got (1, 2) and (1, 2)"
        );
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L306C9-L306C66
    #[test]
    fn test_forward_emissions_last_dimension_not_equal_to_number_of_tags() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        let emissions = Tensor::randn(0.0_f32, 1., (1, 2, 3), &device)?;
        let tags = Tensor::zeros((1, 2), DType::I64, &device)?;
        let crf = make_crf(10, false, None, None, None, &device)?;
        let result = crf.forward(&emissions, &tags, None, Reduction::default());
        println!("{:?}", result);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "expected last dimension of emissions is 10, got 3"
        );
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L315C9-L315C47
    #[test]
    fn test_forward_first_timestep_mask_is_not_all_on() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        let emissions = Tensor::randn(0.0_f32, 1., (3, 2, 4), &device)?;
        let tags = Tensor::zeros((3, 2), DType::I64, &device)?;
        let mask = Tensor::new(&[[1_u8, 1, 1], [0, 0, 0]], &device)
            .unwrap()
            .transpose(0, 1)?;
        let crf = make_crf(4, false, None, None, None, &device)?;

        let result = crf.forward(&emissions, &tags, Some(&mask), Reduction::default());
        println!("{:?}", result);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "mask of the first timestep must all be on"
        );

        let emissions = emissions.transpose(0, 1)?;
        let tags = tags.transpose(0, 1)?;
        let mask = mask.transpose(0, 1)?;
        let crf = make_crf(4, true, None, None, None, &device)?;

        let result = crf.forward(&emissions, &tags, Some(&mask), Reduction::default());
        println!("{:?}", result);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "mask of the first timestep must all be on"
        );
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L345
    fn decode_works_with_mask(dtype: DType, device: &Device) -> Result<()> {
        let crf = make_crf(
            5,
            false,
            Some(
                Tensor::new(&[0.0548, -0.0239, -0.0291, -0.0208, 0.0665], device)?
                    .to_dtype(dtype)?,
            ),
            Some(
                Tensor::new(&[-0.0612, -0.0615, -0.0557, 0.0672, 0.0470], device)?
                    .to_dtype(dtype)?,
            ),
            Some(
                Tensor::new(
                    &[
                        [-0.0751, -0.0941, 0.0248, 0.0900, -0.0776],
                        [0.0381, -0.0550, -0.0333, -0.0124, -0.0356],
                        [-0.0383, -0.0910, 0.0914, -0.0330, -0.0119],
                        [0.0358, 0.0513, 0.0013, -0.0380, 0.0626],
                        [-0.0168, 0.0871, 0.0489, 0.0019, -0.0548],
                    ],
                    device,
                )?
                .to_dtype(dtype)?,
            ),
            device,
        )?;

        let emissions = Tensor::new(
            &[
                [
                    [1.8238, 1.3041, -0.0845, 1.3981, 0.1027],
                    [1.1092, -0.1616, 1.9770, -1.6850, -1.4289],
                ],
                [
                    [0.2831, 0.0936, -1.1957, 0.2637, -0.8048],
                    [0.4553, -0.0393, 2.3307, -0.3505, -2.3531],
                ],
                [
                    [1.6232, 0.2230, 0.3585, -0.7957, -0.2464],
                    [-0.3805, 0.3646, -1.0142, -1.2563, -0.6568],
                ],
            ],
            device,
        )?
        .to_dtype(dtype)?;

        let mask = Tensor::new(&[[1_u8, 1, 1], [1, 1, 0]], device)?.transpose(0, 1)?;
        let best_tags = crf.decode(&emissions, Some(&mask))?;
        println!("best_tags: {:?}", best_tags);
        assert_eq!(best_tags, vec![vec![0, 3, 0], vec![2, 2]]);
        Ok(())
    }

    #[test]
    fn test_decode_works_with_mask() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        for dtype in OK_TYPES {
            decode_works_with_mask(dtype, &device)?;
        }
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L372
    fn decode_works_without_mask(dtype: DType, device: &Device) -> Result<()> {
        let crf = make_crf(
            5,
            false,
            Some(
                Tensor::new(&[0.0762, 0.0743, 0.0234, -0.0387, -0.0269], device)?
                    .to_dtype(dtype)?,
            ),
            Some(
                Tensor::new(&[0.0102, -0.0137, -0.0149, 0.0700, -0.0701], device)?
                    .to_dtype(dtype)?,
            ),
            Some(
                Tensor::new(
                    &[
                        [-0.0620, -0.0527, 0.0034, 0.0694, -0.0853],
                        [0.0922, -0.0613, -0.0592, 0.0482, 0.0632],
                        [-0.0433, 0.0069, -0.0161, -0.0330, -0.0602],
                        [-0.0649, 0.0047, 0.0593, 0.0733, 0.0203],
                        [0.0997, 0.0007, 0.0938, 0.0427, 0.0823],
                    ],
                    device,
                )?
                .to_dtype(dtype)?,
            ),
            device,
        )?;

        let emissions = Tensor::new(
            &[
                [
                    [0.8913, -0.0355, -1.4378, 0.8390, -0.7296],
                    [1.5530, -1.3165, -0.5769, -0.8085, -0.2610],
                ],
                [
                    [-0.9622, -0.3234, -0.5353, -0.4424, -0.1456],
                    [-0.3844, 0.2524, 1.9393, 0.1217, -1.2519],
                ],
                [
                    [-0.1619, -0.2520, 1.9566, 0.4863, 1.5627],
                    [-0.3999, 1.4914, 1.0620, -0.6408, -0.3032],
                ],
            ],
            device,
        )?
        .to_dtype(dtype)?;

        let best_tags_no_mask = crf.decode(&emissions, None)?;
        println!("best_tags: {:?}", best_tags_no_mask);
        assert_eq!(best_tags_no_mask, vec![vec![0, 4, 2], vec![0, 2, 1]]);
        Ok(())
    }

    #[test]
    fn test_decode_works_without_mask() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16] {
            decode_works_without_mask(dtype, &device)?;
        }
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L384C9-L384C28
    fn decode_batched_decode(dtype: DType, device: &Device) -> Result<()> {
        let crf = make_crf(
            5,
            false,
            Some(
                Tensor::new(&[-0.0489, 0.0460, -0.0924, -0.0722, 0.0736], device)?
                    .to_dtype(dtype)?,
            ),
            Some(Tensor::new(&[0.0843, 0.0344, -0.0996, 0.0944, 0.0622], device)?.to_dtype(dtype)?),
            Some(
                Tensor::new(
                    &[
                        [0.0780, -0.0794, 0.0208, 0.0039, 0.0080],
                        [-0.0923, -0.0359, 0.0103, 0.0550, -0.0029],
                        [0.0628, -0.0787, -0.0256, 0.0554, -0.0969],
                        [0.0655, -0.0055, 0.0718, -0.0275, -0.0994],
                        [-0.0492, -0.0953, 0.0862, 0.0580, 0.0422],
                    ],
                    device,
                )?
                .to_dtype(dtype)?,
            ),
            device,
        )?;

        let emissions = Tensor::new(
            &[
                [
                    [0.7720, 0.9488, 0.6672, 1.8839, -0.6844],
                    [1.6192, 0.2733, 0.8063, -0.0377, -2.3208],
                ],
                [
                    [-0.4374, -1.4631, -0.1330, -0.2155, 1.6044],
                    [0.7017, -1.1525, -1.0692, 0.3463, 0.9816],
                ],
                [
                    [-1.3011, 0.5237, -1.1700, -0.9017, -0.5747],
                    [-1.0040, 0.7791, -0.3735, 0.8300, 1.5138],
                ],
            ],
            device,
        )?
        .to_dtype(dtype)?;

        let mask = Tensor::new(&[[1_u8, 1, 1], [1, 1, 0]], device)?.transpose(0, 1)?;

        let batched = crf.decode(&emissions, Some(&mask))?;
        println!("batched: {:?}", batched);

        let batch_size = 2;
        let mut non_batched = vec![];
        for i in 0..batch_size {
            let emissions = emissions.i((.., i, ..))?.unsqueeze(1)?.contiguous()?;
            let mask = mask.i((.., i))?.unsqueeze(1)?.contiguous()?;

            let result = crf.decode(&emissions, Some(&mask))?;
            non_batched.push(result[0].clone());
        }
        println!("non_batched: {:?}", non_batched);

        assert_eq!(batched, non_batched);
        assert_eq!(batched, vec![vec![3, 4, 1], vec![0, 4]]);
        Ok(())
    }

    #[test]
    fn test_decode_batched_decode() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        for dtype in OK_TYPES {
            decode_batched_decode(dtype, &device)?;
        }
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L408
    fn decode_batch_first(dtype: DType, device: &Device) -> Result<()> {
        let crf = make_crf(
            5,
            false,
            Some(
                Tensor::new(&[-0.0464, 0.0818, 0.0829, -0.0121, -0.0788], device)?
                    .to_dtype(dtype)?,
            ),
            Some(
                Tensor::new(&[-0.0088, 0.0586, 0.0057, 0.0316, -0.0388], device)?
                    .to_dtype(dtype)?,
            ),
            Some(
                Tensor::new(
                    &[
                        [-0.0536, -0.0093, 0.0276, 0.0351, 0.0604],
                        [0.0734, 0.0764, -0.0773, 0.0821, 0.0294],
                        [-0.0540, -0.0158, 0.0437, 0.0992, 0.0473],
                        [0.0875, 0.0324, -0.0941, 0.0585, 0.0761],
                        [-0.0930, -0.0832, 0.0290, 0.0974, 0.0914],
                    ],
                    device,
                )?
                .to_dtype(dtype)?,
            ),
            device,
        )?;

        let emissions = Tensor::new(
            &[
                [
                    [-0.6633, 1.4045, -1.3710, 1.5054, 0.8431],
                    [-0.1157, -0.0201, -0.2685, -0.6683, 0.0213],
                ],
                [
                    [-0.7870, -0.2497, -0.3901, 0.0181, 0.0976],
                    [0.4487, 0.2629, 2.2021, -0.7489, 0.1199],
                ],
                [
                    [0.7837, -0.0174, -0.3873, -0.4722, -0.2462],
                    [-0.6268, -0.9438, 0.6666, -0.6545, 1.0409],
                ],
            ],
            device,
        )?
        .to_dtype(dtype)?;

        let best_tags = crf.decode(&emissions, None)?;
        println!("best_tags: {:?}", best_tags);

        let crf_bf = make_crf(
            5,
            true,
            Some(crf.start_transitions),
            Some(crf.end_transitions),
            Some(crf.transitions),
            device,
        )?;

        let emissions = emissions.transpose(0, 1)?;
        let best_tags_bf = crf_bf.decode(&emissions, None)?;
        println!("best_tags_bf: {:?}", best_tags_bf);

        assert_eq!(best_tags, best_tags_bf);
        assert_eq!(best_tags, vec![vec![1, 3, 0], vec![1, 2, 4]]);
        Ok(())
    }

    #[test]
    fn test_decode_batch_first() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        for dtype in OK_TYPES {
            decode_batch_first(dtype, &device)?;
        }
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L427
    #[test]
    fn test_decode_emissions_has_bad_number_of_dimension() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        let emissions = Tensor::randn(0.0_f32, 1., (1, 2), &device)?;
        let crf = make_crf(5, false, None, None, None, &device)?;
        let result = crf.decode(&emissions, None);
        println!("{:?}", result);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "emissions must have 3 dimensions, got 2"
        );
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L435
    #[test]
    fn test_decode_emissions_last_dimension_not_equal_to_number_of_tags() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        let emissions = Tensor::randn(0.0_f32, 1., (1, 2, 3), &device)?;
        let crf = make_crf(10, false, None, None, None, &device)?;
        let result = crf.decode(&emissions, None);
        println!("{:?}", result);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "expected last dimension of emissions is 10, got 3"
        );
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L443
    #[test]
    fn test_decode_emissions_and_mask_size_mismatch() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        let emissions = Tensor::randn(0.0_f32, 1., (1, 2, 3), &device)?;
        let mask = Tensor::new(&[[1_u8, 1], [1, 0]], &device)?;
        let crf = make_crf(3, false, None, None, None, &device)?;
        let result = crf.decode(&emissions, Some(&mask));
        println!("{:?}", result);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "the first two dimensions of emissions and mask must match, got (1, 2) and (2, 2)"
        );
        Ok(())
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/tests/test_crf.py#L454
    #[test]
    fn test_decode_first_timestep_mask_is_not_all_on() -> Result<()> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let device = use_gpu(true)?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = use_gpu(false)?;

        let emissions = Tensor::randn(0.0_f32, 1., (3, 2, 4), &device)?;
        let mask = Tensor::new(&[[1_u8, 1, 1], [0, 0, 0]], &device)?.transpose(0, 1)?;

        let crf = make_crf(4, false, None, None, None, &device)?;
        let result = crf.decode(&emissions, Some(&mask));
        println!("{:?}", result);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "mask of the first timestep must all be on"
        );

        let emissions = emissions.transpose(0, 1)?;
        let mask = mask.transpose(0, 1)?;
        let crf = make_crf(4, true, None, None, None, &device)?;
        let result = crf.decode(&emissions, Some(&mask));
        println!("{:?}", result);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "mask of the first timestep must all be on"
        );
        Ok(())
    }
}
