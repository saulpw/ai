#!/usr/bin/env python3

'''
Usage:
    $0 chat <msg>
        Send <msg> directly to GPT and print its response.

    $0 embed <filenames...>
        Chunk and index sections in filenames delimited by '\\n\\n'.

    $0 query "<question>"
        Find top documents that match <question> and send them to GPT along with <question>, and print its response.
'''

import os
import sys
import json

fn_embeddings = 'embeddings.jsonl'
context_len = 8191  # from completion model
cfg_topn = 5  # number of search results to use
cfg_chatmodel = 'gpt-3.5-turbo'

def stderr(x):
    if isinstance(x, dict):
        x = ' '.join(f'{k}={v}' for k, v in x.items())
    print(x, file=sys.stderr)


def user(s):
    return dict(role='user', content=s)

def system(s):
    return dict(role='system', content=s)


def windowchunk(text, window_size=1000, sep='\n\n'):
    win = []
    tot = 0
    for i, unit in enumerate(text.split(sep)):
        n = len(unit)
        if tot+n > window_size:
            if win:
                yield i, sep.join(win)
                win = []
                tot = 0

        win.append(unit)
        tot += n

    if win:
        yield i, sep.join(win)


def dot_product(a, b):
    return sum([x*y for x,y in zip(a,b)])


class VectorStore:
    def __init__(self, fn):
        self.fn = fn
        self.embeddings = dict()
        try:
            for line in open(fn):
                emb = json.loads(line)
                self.embeddings[emb['text']] = emb
        except FileNotFoundError as e:
            pass

    def get(self, text, **kwargs):
        if text in self.embeddings:
            return self.embeddings[text]

        import openai

        resp = openai.Embedding.create(input=text, model='text-embedding-ada-002')
        emb = dict(
            text=text,
            embedding=resp['data'][0]['embedding'],
            **kwargs
        )

        embstr = json.dumps(emb)
        print(embstr, file=open(self.fn, 'a'))

        return emb

    def find(self, needle, n=1):
        yield from sorted([
            (dot_product(needle['embedding'], emb['embedding']), emb)
                for emb in self.embeddings.values()
                    if emb.get('source', '')
        ], reverse=True)[:n]


def main_embed(*args):
    vstore = VectorStore(fn_embeddings)
    for fn in args:
        print(fn, file=sys.stderr)
        chunk_size = context_len/2/cfg_topn
        contents = open(fn).read()
        for line, w in windowchunk(contents, chunk_size, sep='\n\n'):
            emb = vstore.get(w, source=f'{fn}:{line}')


def main_query(*args):
    q = ' '.join(args)
    if not q:
        q = sys.stdin.read()

    vstore = VectorStore(fn_embeddings)
    qemb = vstore.get(q, source='')

    similar = []
    for similarity, emb in vstore.find(qemb, n=cfg_topn):
        if similarity > 0.10:
            similar.append(emb)
            print(f"{similarity:.02f} {emb['source']}")

    msgs = [system('You are a helpful assistant. This is documentation pertaining to the user query:')]
    msgs.extend(user(emb['text']) for emb in similar)
    msgs = [system('Answer this user query about VisiData precisely and concisely, using the above documentation:')]
    msgs.append(user(q))

    import openai
    resp = openai.ChatCompletion.create(messages=msgs, model=cfg_chatmodel)
    stderr(resp['usage'])
    stderr(resp['usage'])

    return resp['choices'][0]['message']['content']


def main_chat(*args):
    import openai
    q = ' '.join(args)
    resp = openai.ChatCompletion.create(messages=[user(q)], model=cfg_chatmodel)
    stderr(resp['usage'])

    return resp['choices'][0]['message']['content']


def main_multichat(*args):
    import openai
    q = ' '.join(args)
    resp = openai.ChatCompletion.create(messages=[user(q)], model=cfg_chatmodel)
    stderr(resp['usage'])

    return resp['choices'][0]['message']['content']


def main_help(*args):
    return __doc__


def main(cmd='', *args):
    f = globals().get('main_'+cmd, None)
    if not f:
        stderr(f'No command "{cmd}"')
        stderr(main_help())
        return

    r = f(*args)
    if r:
        print(str(r))


if __name__ == '__main__':
    main(*sys.argv[1:])
