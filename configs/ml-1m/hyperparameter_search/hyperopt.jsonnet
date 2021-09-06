{
  module : {
    model: {
      num_transformer_heads: {
        hyper_opt: {
          suggest: "int",
          params: {
            low: 2,
            high: 2,
            step: 2
          }
        }
      },
      num_transformer_layers: {
        hyper_opt: {
          suggest: "int",
          params: {
            low: 2,
            high: 8,
            step: 2
          }
        }
      },
      transformer_hidden_size: {
        hyper_opt: {
          suggest: "int",
          params: {
            low: 2,
            high: 8,
            step: 2
          },
          dependency: {
            type: "multiply",
            on: "module.model.num_transformer_heads"
          }
        }
      },
      transformer_dropout: 0.1,
      nonlinearity: {
        hyper_opt: {
          suggest: "categorical",
          params: {
            choices: ['relu', 'tanh']
          },
          dependency: {
            on: "module.model.num_transformer_heads",
            type: "optimize_iff",
            conditions: [{
              type: 'equal',
              compare_value: 5
            }]
          }
        }
      }
    }
  }
}