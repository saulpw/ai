
# GPT Discord bot

## CLI Usage

    ./ai.py chat <msg>
        Send <msg> directly to GPT and print its response.

    ./ai.py embed <filenames...>
        Chunk and index sections in filenames delimited by '\\n\\n'.  save to embeddings.jsonl

    ./ai.py query <question>
        Find top documents that match <question> and send them to GPT along with <question>, and print its response.

## Discord Usage

- mention `VisiData` in its presence, or start a message with `!', and it will find 5 relevant embedded documents and send them via the OpenAI API to GPT, and reply with its response.

