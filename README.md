# docutranslate

does the following:

scanned pdf file -> images -> text -> gpt-4o -> translated word doc

see [test.ipynb](/test.ipynb) for details

## example usage

install requirements

```bash
pip install -r requirements.txt
```

process the entire PDF:

```bash
python main.py attention.pdf --language "Chinese (Traditional)"
```

process a single page:

```bash
python main.py attention.pdf --language "Chinese (Traditional)" --single-page --page-number 1
```

## references

- [Introduction to gpt-4o | OpenAI Cookbook](https://cookbook.openai.com/examples/gpt4o/introduction_to_gpt4o?curius=2055)
