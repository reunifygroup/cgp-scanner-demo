One suggestion: Your augmentations are quite gentle (which is good for classification). For embeddings, you could be slightly more
aggressive since we're doing similarity matching. But let's test with current settings first - they should work well.

---

`source .venv/bin/activate`

`pip freeze > requirements.txt`
`pip install -r requirements.txt`

TODO:

-   DONE - ui with actual validation by similarity
-   DONE - (zip embeddings and removed pipeline) have a way to deploy without pipeline, passing to vercel maybe a env value "run pipeline true/false"
-   split the validation on server side w json
-   add OCR client side -> get all random read, hoping for the name of the card -> check if response name (top 3) is contained in the OCR
-   use postgres w/o json
