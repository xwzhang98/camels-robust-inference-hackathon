# Submitting

You submit code and the organizers run your `predict.py` on it.

## How

1. **Fork** this repository (once per team).
2. Add one directory, `submissions/<team_name>/`, containing:

```
   predict.py     defines load_model(model_dir) and predict(model, catalog_path)
   model.pt       your weights, plus any helper modules predict.py imports
   README.md      a few lines: team members, what the model is, what you tried
   gnn.py.        defines your model
   other helper functions (such as dataloader, postprocessing)
```

   The easiest start is to copy `baseline/gnn/` and edit it. Keep the directory under 50 MB;
   if your weights are bigger, put a download link in the README and tell us.

3. Open a **pull request** to `main`, titled `[<team_name>]`. Push more commits to the same
   branch to update it — **the last commit before the deadline is what gets scored.**


## Deadline

**Thu <time>.** Open your PR early and keep pushing to it.
