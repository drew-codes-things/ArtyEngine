# Stores in the older layout

The commands are in the [README](../README.md#stores-in-the-older-layout). This page says what each one does.

`plan` lists every model name, where its manifest will go and which old copies stay behind, and changes nothing.
`migrate` writes one catalog manifest per name (with the old manifest's time, so `/api/tags` keeps its order), keeps
a copy of the old folders in the backup folder, then moves the old folders and the old manifest documents in `blobs/`
there. Model files are never touched, copied or renamed. Short names stay short, and a name with a namespace keeps
it (`someone/model:tag`); a model from another registry keeps that registry's host in its name. The one registry of
an old folder becomes the short names. When a folder holds several, or when you want to choose, `--default-host`
names it; once given, only the hosts given become short names in every folder, so give it once per old folder (`plan`
adds a note for a folder it leaves out). Two old names that would become one are refused, and a name in both old
folders counts as an older copy only when both hold the same model file; an older copy that can't be read stays in
the backup unread. A model whose old manifest can't be converted, or whose two copies hold different model files,
stops the run until it is left out with `--skip <name>`. The backup folder must be empty and reachable by a rename
(the same mount). Every command that changes the store refuses while anything accepts a connection on `--engine-url`
(`http://127.0.0.1:11438` unless given, `http://` added when the address has no scheme, `none` to skip), and when it
can't tell; a stalled engine counts as running.

`undo` first checks that the backup can bring the old layout back (from `moved/`, or from `copy/` when `moved/` is
gone) and changes nothing when it can't; then it moves everything back, removes the catalog manifests the migration
wrote and checks the old layout byte for byte against the record taken before the migration. It refuses when a
migrated manifest changed or a model was pulled since, or when the backup was already undone and a later migrate
holds the store, and names any model whose blobs were deleted in between or whose old manifest can't be read. An undo
that stopped while copying can be run again. A backup made before 15 Nov 2026 needs its `moved/` folder: its record
does not say whether the run finished, so its `copy/` alone is not trusted. `tidy` moves the old manifest documents
of a store migrated by an earlier version of this tool into its backup (and finishes a move that stopped); `untidy`
puts them back. Both refuse once the migration was undone. Empty probe folders (`.migrate-probe-*`) left by a stopped
run are removed by the next migrate or undo. At startup the engine names every root that is still in the older layout.

## Unused blobs

`tools/unused_blobs.py --store <store>` lists what no catalog manifest names: partial downloads, model files, old
manifest documents and anything else in `blobs/`, each with its kind and size. `--remove` deletes the list once it
is printed, and refuses (exit 2, nothing removed) while anything listens on `--engine-url` (the engine's address
unless given, `none` to skip), while the store still holds an old manifest folder, has no catalog manifest, holds the
leftovers of a stopped migration, undo or manifest write (the refusal names which; remove it by hand if nothing is
running), or lists old manifest documents (use `tidy` for those, so undo can bring them back). Plain `--remove`
deletes only partial downloads; model files and other blobs no manifest names, such as the weights, template, system
prompt or parameters of a model left out of a migration with `--skip` or of an older copy in its backup, go only with
`--remove-unnamed-blobs` as well. Never run it on a store in the older layout: there every blob counts as unused.
