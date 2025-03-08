#TODO

## Documentations
- [ ] Add logit lens to the docs
- [ ] Update the docs string
- [ ] Extend the two tutorials

## Features
- [ ] Add `extract_head_keys_projected` and `extract_head_queries_projected` in the config
- [ ] Extent the module wrapper to be able to handle multiple modules of the same models (now support only attention)

## Bugs

## Refactorings
- [ ] Handling better the shape of the returned tensors in the ActivationCache
- [ ] (low priority) Refactor the python modules to a more intuitive structure

## Tests
- [ ] Expand test for patching
- [ ] Expand test for the ablation
- [ ] Expand test for generation
- [ ] Expand test for edge cases in token index 
- [ ] Expand test for edge cases in aggregation of ActivationCache