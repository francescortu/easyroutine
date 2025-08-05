# TODO

## Documentations
- [x] Add logit lens to the docs
- [x] Add activation saver to docs + add docstring for activation saver
- [ ] Update the docs string
- [x] Extend the three tutorials

## Features
- [ ] HIGH PRIORITY: Check and test the `intervention_query_key_value_hook` in the interventions.py file. It should be used to handle the query, key, and value hooks in a more generic way.
- [ ] HIGH PRIORITY:  Fix the intervention to work during with generation also with `use_cache=True`. 

- [ ] implement logger that do not print two times the warning message (wanning_once). Update all the loggers



## Bugs

## Refactorings
- [ ] Find a better way to handle the slicing and the gradient keeping (now, if i keep the gradient, i cannot slice it in the hooks and I should do after. It will be great to have a way to slice the tensor in the hook and keep the gradient or have a clean logic to handle the slicing post hook. An idea could be to slice inside activation cache)
- [ ] Refactoring the `HookedModel.compute_patching()` in a cleaner way.
- [ ] Handling better the shape of the returned tensors in the ActivationCache.
- [ ] On ActivationCache, when switching back to list, ensure to unpack all the tensors already in the cache

- [ ] (low priority) Refactor the python modules to a more intuitive structure

## Tests
- [ ] HIGH PRIORITY: Add test for `llava-onevision` model
- [ ] Expand test for intervention
- [ ] Add test for local hooks
- [ ] Expand test for generation
- [ ] Expand test for edge cases in token index 
- [ ] Expand test for edge cases in aggregation of ActivationCache
- [ ] Test ActivatioNCache.map_to_dict()

## Console
- [ ] Implement a rich.console module to handle output for the entire package
- [ ] Substitute the current print statements with the rich.console module