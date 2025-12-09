One suggestion: Your augmentations are quite gentle (which is good for classification). For embeddings, you could be slightly more
aggressive since we're doing similarity matching. But let's test with current settings first - they should work well.

---

`source .venv/bin/activate`

`pip freeze > requirements.txt`
`pip install -r requirements.txt`
