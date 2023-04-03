{
  module: {
    learning_rate: {
      hyper_opt: {
        suggest: "categorical",
        params: {
          choices: [0.1, 0.01, 0.001]
        }
      }
    },
    model: {
      num_transformer_heads: {
        hyper_opt: {
          suggest: "int",
          params: {
            low: 2,
            high: 4,
            step: 1
          }
        }
      },
      num_transformer_layers: {
        hyper_opt: {
          suggest: "int",
          params: {
            low: 2,
            high: 4,
            step: 1
          }
        }
      },
      transformer_hidden_size: {
        hyper_opt: {
          suggest: "int",
          params: {
            low: 4,
            high: 8,
            step: 1
          },
          dependency: {
            type: "multiply",
            on: "module.model.num_transformer_heads"
          }
        }
      },
      transformer_dropout: {
        hyper_opt: {
          suggest: "float",
          params: {
            low: 0.1,
            high: 0.5,
            step: 0.1
          }
        }
      }
    }
  }
}