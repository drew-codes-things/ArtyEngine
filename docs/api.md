# The engine's HTTP API

| Endpoint | In the engine |
|---|---|
| `GET /`, `HEAD /` | Answers that the engine is running |
| `GET /api/version` | The engine's own version |
| `GET /api/tags` | Every model in the store and in `model_roots`, minus `internal_models`, then the `remote/` models of `remote_models` |
| `GET /api/ps` | Loaded models, their context and VRAM |
| `POST /api/show` | Details, parameters, template, capabilities, model info, and `runnable` |
| `POST /api/chat` | Streamed or not; tools; `think` (true, false, unset, a level); `format` (`"json"`, a schema); images; `keep_alive`; sampling; `num_ctx`; `num_gpu: 0`; `_debug_render_only` |
| `POST /api/generate` | Prompts, streamed and not; an empty prompt loads a model, and with `keep_alive: 0` unloads it; `raw` prompts |
| `POST /api/embed`, `POST /api/embeddings` | Embeddings |
| `POST /api/pull` | Hugging Face names (`hf.co/<owner>/<repo>:<quant>`) and the short names in `arty_engine/short_names.json`, or a short name with its own `source`, with progress; one pull at a time (more below) |
| `DELETE /api/delete` | Models in the engine's own store. Models read in place from `model_roots` answer 405, internal models and deletes during a pull answer 409 |
| `POST /api/generate` with `suffix` | **501** |
| `POST /api/generate` raw with images or a `format` | **501** |
| `POST /api/create`, `/api/copy`, `/api/push` | **501** |
| Models whose file keeps the vision weights inside it (such as `qwen2.5vl:7b`) | Listed, but chats answer **501**: not supported, pull a build with a separate projector file instead |
| Split GGUF files on Hugging Face | Refused by `/api/pull` |
| Anything else (`/api/blobs`, the `/v1` OpenAI-style routes) | 404 |

Every answer carries `Server: arty-engine/<version>`.

## Models the engine cannot read

A model the engine can't read or describe is left out of `/api/tags`, and a loaded one it can't describe out of
`/api/ps`, with a warning that names it, so one bad file never takes the lists away from the other models; the chat
and show of a model it can't read answer `couldn't read <name>: ...`. A model
file's header is read within limits far above any real model: 65,536 keys, 64 KiB for a key or a tensor name, 64 MiB
for one text value and 128 MiB of text in all (keys, tensor names, text values and the text in short arrays; the long
token and merge lists are stepped over and not counted, so a large vocabulary reads), 8 dimensions for a tensor, arrays
nested at most 8 deep, and a million values and tensors in all. Tensor names are read but not kept. A file the reader
refuses is not read again until it is replaced or its size or change time changes. A header count of the wrong type
(such as a context length written as text) or a float too large for a 64-bit whole number counts as unknown, and a
number JSON can't hold (not a number, infinity) and a long list inside a short one are shown as `null` in
`model_info`.

## Thinking that leaks into the answer

Thinking that leaks into the answer: some models (seen with `mimo-v2.6:9b`) write their reasoning and a closing
`</think>` into the reply even with `think` false. Whether a model can leak is decided once when it loads: a thinking
model can when it is named in `thinking_leakers` or its chat template does not read `enable_thinking`. `mimo-v2.6:9b`
needs naming: its template does read the switch, yet the model still writes its reasoning into the answer. Any other
model (such as `qwen3:14b`) streams its content at once and exactly as written, think tags included. For a model that
can leak, chat and generate replies (not raw ones) are read the same way streamed and not streamed, so both give the
same answer. Until the first closing tag, reasoning from the model or the end of the reply, content is held back, so a
streamed reply without a closing tag arrives in one piece at the end, however long it is. The text before the first
closing tag goes into `thinking`, or is dropped when the request has `think` false, and the answer starts after it,
without the blank lines that follow the tag. After that, a `<think>` ... `</think>` block in the answer is a second
block of reasoning and goes into `thinking` too, a lone closing tag is taken out, and both are logged as a warning; an
opening tag there that is never closed stays in the answer as written, unless the reply was cut by the length limit,
when the text after it is thinking. The same characters mean both things, so an answer from such a model that writes
the tags itself keeps only what follows the first closing tag, without any later tagged block. A reply with no closing
tag at all is delivered exactly as written, unless it was cut by the length limit after an opening tag: then the text
after the tag is thinking, the words before it (if any) are the answer, and a reply with words before the tag is
logged as a warning. A reply cut by the length limit before any tag can't be told from a cut answer, so it is
delivered as written and logged as a warning, since it may be reasoning. Once the model's runner has delivered
reasoning of its own, a model that is held only because its template does not read `enable_thinking` has its answer
passed on exactly as written; one named in `thinking_leakers` still has tagged blocks taken out of the answer. A name
in `thinking_leakers` that matches no installed model is logged as a warning at start. A reply from which the engine
moved text to `thinking` while the runner gave no reasoning of its own is logged at the normal level as `leak caught:`
with the model's name, each rule that moved text (text before the first closing tag, a tagged block in the answer, or
text after an opening tag in a reply cut by the length limit) with how many times it moved text and how many
characters of reasoning it moved (tags and surrounding blank space not counted), and the number of that model's
replies with a caught leak since the engine started. It is logged before the reply's last line is sent, and also when
the reader hangs up or the runner fails after the move. Tagged blocks taken out after the runner's own reasoning are
not counted (they are logged as late tags). The count is of replies the engine edited, not of proven leaks: an answer
of a held model that writes the tags itself, with no reasoning from the runner, cannot be told apart from a leak and
is counted too, so among the leaks the engine can see, the count is an upper bound. A tag with no reasoning in front
of it moves nothing and is not counted. Two kinds of leak are outside the count: a reply cut by the length limit
before any tag, which has its own warning, and a stream that ends with no finish reason and no tag, which is
delivered as it stands with no line; to judge a model, read `leak caught` together with the tagless-cut warnings. The
count starts again at 0 when the engine restarts.

## Pulls

Pulls: a short name already in the store is not downloaded again; one the engine can't run or read answers 409
(delete it, then pull). Each short name is pinned to the sha256 of its files, so a file changed on Hugging Face is
refused (502), and a short name that pins no projector takes none. A file listed without a size is refused (502).
Pulled manifests record the repo and revision. Any other name answers 400 with the form to use. With
`"stream": false` a pull that would break `pull_keep_free_mb` answers 507; a streamed pull reports that and a pin
mismatch inside the stream.

A pull can also name its own source: `{"model": "qwen2.5vl:7b", "source": "hf.co/<owner>/<repo>:Q4_K_M"}` fetches that
repository and quant and installs it under the short name given, and the manifest records the source. Without `source`
the short names in `arty_engine/short_names.json` decide, as above. Pulling a name already installed from the same
source downloads nothing; `hf.co` and `huggingface.co` are the same host, and the owner, repository and quant are
compared without regard to case. From a different source the new files are downloaded and checked first, and only then
does the new manifest replace the old one, and the old copy is unloaded. A replace keeps only the new weights: the old
copy's template, system prompt and parameters are not carried over (the status line names the ones dropped), and the
old copy's files stay in the store, since a migration backup or a model left out with `--skip` may still need them.
The status line says how much space they hold (their model, settings and config files) and prints the cleanup command
in full, with the store's path and the engine's address; it runs only with the engine stopped, and it removes every
blob no model names, not only the old copy's, so run it without `--remove` first to see the list. Until then the old
files count against `pull_keep_free_mb`, so a second replace of a large model may answer 507. Like any pull that
writes a manifest, a replace means a migration made before it can no longer be undone. A new source whose files are
the old copy's is not a replace: only the recorded source changes, nothing is unloaded, and the loaded copy keeps
answering (the recorded source is not part of what decides whether a loaded model can be reused), and `/api/ps` shows
the new digest at once. A name installed only in another store root gets its new copy in the engine's own store and
the other root's copy is left as it is, no longer used. Every pull that names a source is logged as a warning with the
caller's address, the name and the source, and every replace with the old and the new source; `/api/show` shows a
model's recorded source (`name` and `revision`) when it has one. Projector rules, size checks and `pull_keep_free_mb`
apply as for any pull, and a source that matches a short name's catalogue entry, in any of the spellings above, keeps
its sha256 pins. `source` must be a Hugging Face name (400 otherwise), only a short name or that same Hugging Face
name can take it (400), and an internal model's name answers 409.
