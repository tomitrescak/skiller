# Prerequisties

- Python 3.10

# Instructions

1. Create the virtual environment

```
python3 -m venv .venv
```

2. Activate it

```
source .venv/bin/activate
```

3. Install Requirements

```
pip install -r requirements.txt
```

4. Create Embeddings

You need to pre-cache skill framework embeddings.
This will take a long time, but you only run this once.

```
python -m main --init
```

5. Have Fun!

You can now run skill extraction with:

```
python -m main --input test.txt --threshold 0.3
```

Just put the text from which you want to extract skills into the file "test.txt"
