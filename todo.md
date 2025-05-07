# TODO

## Documentations
- [x] Add logit lens to the docs
- [x] Add activation saver to docs + add docstring for activation saver
- [ ] Update the docs string
- [x] Extend the three tutorials

## Features
- [ ] Add support for keep gradient for all the keys in the activation cache
- [x] Add the metadata in the cache and simply the ActivationSaver if it save a cache object
- [x] Extend the intervention to support all the keys 
- [x] Add `extract_head_keys_projected` and `extract_head_queries_projected` in the config
- [x] Extent the module wrapper to be able to handle multiple modules of the same models (now support only attention)

## Bugs

## Refactorings
- [ ] Refactoring the `HookedModel.compute_patching()` in a cleaner way.
- [ ] Handling better the shape of the returned tensors in the ActivationCache.

- [ ] (low priority) Refactor the python modules to a more intuitive structure

## Tests
- [ ] Expand test for intervention
- [ ] Add test for local hooks
- [ ] Expand test for generation
- [ ] Expand test for edge cases in token index 
- [ ] Expand test for edge cases in aggregation of ActivationCache
- [ ] Test ActivatioNCache.map_to_dict()

## Console
- [ ] Implement a rich.console module to handle output for the entire package
- [ ] Substitute the current print statements with the rich.console module