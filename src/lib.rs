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

        // println!("tags: {:?}", tags.to_vec2::<i64>()?);

        let mask = mask.to_dtype(emissions.dtype())?;

        // println!("mask: {:?}", mask.to_vec2::<f32>()?);

        // println!(
        //     "start_transitions: {:?}",
        //     self.start_transitions.to_vec1::<f32>()?
        // );

        let mut score = self.start_transitions.i(&tags.i(0)?)?;
        // println!("score: {:?}", score.to_vec1::<f32>()?);

        // println!("emissions: {:?}", emissions.to_vec3::<f32>()?);

        let z = multi_index(&emissions.i((0, 0..batch_size))?, &tags.i(0)?)?;
        // println!("z: {:?}", z.to_vec1::<f32>()?);

        score = score.broadcast_add(&z)?;
        // println!("score: {:?}", score.to_vec1::<f32>()?);

        for i in 1..seq_length {
            let z = multi_index(&self.transitions.i(&tags.i(i - 1)?)?, &tags.i(i)?)?;
            // println!("{i}, z: {:?}", z.to_vec1::<f32>()?);
            score = score.broadcast_add(&z.broadcast_mul(&mask.i(i)?)?)?;

            let z = multi_index(&emissions.i((i, 0..batch_size))?, &tags.i(i)?)?;
            score = score.broadcast_add(&z.broadcast_mul(&mask.i(i)?)?)?;
        }

        let seq_ends = mask
            .to_dtype(DType::I64)?
            .sum(0)?
            .broadcast_sub(&Tensor::ones(1, DType::I64, mask.device())?)?;

        let last_tags = multi_index(
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
        // println!("score: {:?}", score.to_vec2::<f32>()?);
        // println!("result: {:?}", score.log_sum_exp(1)?.to_vec1::<f32>()?);
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
        // println!("score: {:?}", score.to_vec2::<f32>()?);

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

pub(crate) fn max_indices<D: Dim + Copy>(x: &Tensor, dim: D) -> Result<(Tensor, Tensor)> {
    let max = x.max(dim)?;
    let idx = x.argmax(dim)?;
    Ok((max, idx))
}

// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L60
    #[test]
    fn test_minial() {
        let num_tags = 10;
        let crf = CRF::new(num_tags, false, &Device::Cpu).unwrap();
        assert_eq!(crf.num_tags, num_tags);
        assert!(!crf.batch_first);
        assert_eq!(crf.start_transitions.dims1().unwrap(), num_tags);
        assert_eq!(crf.end_transitions.dims1().unwrap(), num_tags);
        assert_eq!(crf.transitions.dims2().unwrap(), (num_tags, num_tags));
        println!("crf:{}", crf);
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L74
    #[test]
    fn test_full() {
        let crf = CRF::new(10, true, &Device::Cpu).unwrap();
        assert!(crf.batch_first);
    }

    /// https://github.com/kmkurn/pytorch-crf/blob/623e3402d00a2728e99d6e8486010d67c754267b/tests/test_crf.py#L78C9-L78C34
    #[test]
    fn test_nonpositive_num_tags() {
        let crf = CRF::new(0, false, &Device::Cpu);
        assert!(crf.is_err());
    }
}
