# Security notes

The engine listens on `127.0.0.1:11438` and has no authentication of its own. Keep it on loopback, or put a reverse
proxy with authentication in front of it. Requests that carry an `Origin` header from a web page other than
localhost are refused with 403, and so are requests for a `Host` other than localhost while the engine listens on
loopback, which stops other web sites from reaching it through your browser. A proxy that passes its own host name,
or a web app you trust, goes in `allowed_hosts` or `allowed_origins`.

The shipped service unit runs the engine as root (see [Installing](install.md)).
