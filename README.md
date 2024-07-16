# Candle CRF

Candle CRF ports [pytorch-crf](https://github.com/kmkurn/pytorch-crf) to [Huggingface Candle](https://github.com/huggingface/candle).

## Initialization

- Pytorch CRF

    ```python
    crf = CRF(num_tags, batch_first=false)
    ```

- Candle CRF

    ```rust
    let crf = CRF::new(num_tags, false, &candle_core::Device::Cpu).unwrap();
    ```

## Forward

- Pytorch CRF

    ```python
    llh = crf(emissions, tags, mask)
    ```

- Candle CRF

    ```rust
    let llh = crf
            .forward(&emissions, &tags, Some(&mask), Reduction::default())
            .unwrap();
    ```

## Decode

- Pytorch CRF

    ```python
    best_tags = crf.decode(emissions, mask)
    ```

- Candle CRF

    ```rust
    let best_tags = crf.decode(&emissions, Some(&mask)).unwrap();
    ```
