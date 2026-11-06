"""Deployment profiles: where a service runs, where its data and state live, and which providers it may load.

A profile is a JSON file (``profiles/*.json`` holds the local, Reef, Bruno and AWS templates). ``${VAR}`` and
``${VAR:-default}`` are expanded from the environment, so site paths, buckets and URLs stay out of the profile.
Credentials are never written in a profile: object-store access uses the environment of the process (for example
``AWS_PROFILE`` or an instance role), and keys that look like secrets are refused.

    python -m cellcanvas_service.profiles check profiles/reef.json [CONFIG]

validates a profile and, with a service config, resolves every dataset location without reading voxel data.

Dataset paths in the service config that are relative are resolved against ``data.root``; absolute paths and
URLs are kept. ``data.kind`` is ``posix`` (local disk or a shared filesystem) or ``fsspec`` (an object store such as
``s3://``), whose stores are opened with ``data.storage_options``. Writable stores (painting, prediction) and the
state directory should be local or a shared filesystem; image and features may be remote and read-only.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

FIELDS = {"name", "description", "bind", "state_dir", "config", "data", "compute", "providers",
          "prompt_backends"}
SECRET = re.compile(r"secret|password|token|credential|access_key|key_id|session", re.IGNORECASE)
VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
READ_ONLY = {"image_path": True, "features_path": True, "labels_path": False, "segmentation_path": False}


class ProfileError(ValueError):
    pass


def expand(value, environ):
    if isinstance(value, str):
        def replace(match):
            name, default = match.group(1), match.group(2)
            if name in environ:
                return environ[name]
            if default is not None:
                return default
            raise ProfileError(f"Environment variable {name} is not set")

        return VARIABLE.sub(replace, value)
    if isinstance(value, list):
        return [expand(v, environ) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, environ) for k, v in value.items()}
    return value


def load_profile(path, environ=None):
    """Read, expand and validate a deployment profile."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or not raw.get("name"):
        raise ProfileError("A profile is a JSON object with a name")
    unknown = set(raw) - FIELDS
    if unknown:
        raise ProfileError(f"Unsupported profile fields {sorted(unknown)}")
    profile = expand(raw, os.environ if environ is None else environ)
    bind = profile.setdefault("bind", {})
    bind.setdefault("host", "127.0.0.1")
    bind["port"] = int(bind.get("port", 8001))
    exposure = bind.setdefault("exposure", "loopback")
    if exposure not in ("loopback", "behind_proxy"):
        raise ProfileError("bind.exposure must be loopback or behind_proxy")
    if bind["host"] not in LOOPBACK and exposure != "behind_proxy":
        raise ProfileError("The service has no authentication; bind a non-loopback host only with "
                           "exposure=behind_proxy, behind an authenticating proxy")
    data = profile.setdefault("data", {})
    data.setdefault("kind", "posix")
    if data["kind"] not in ("posix", "fsspec"):
        raise ProfileError("data.kind must be posix or fsspec")
    options = data.setdefault("storage_options", {})
    secrets = sorted(key for key in _keys(options) if SECRET.search(key))
    if secrets:
        raise ProfileError(f"storage_options must not contain credentials ({secrets}); use the process environment")
    if data.setdefault("credentials", "environment") not in ("environment", "anonymous", "none"):
        raise ProfileError("data.credentials must be environment, anonymous or none")
    providers = profile.setdefault("providers", {})
    providers.setdefault("allow_private", False)
    providers.setdefault("plugins", [])
    for backend in profile.setdefault("prompt_backends", []):
        if set(backend) - {"name", "url", "optional"} or not backend.get("name") or not backend.get("url"):
            raise ProfileError("prompt_backends entries need name and url")
    threads = profile.setdefault("compute", {}).setdefault("threads", 1)
    if not isinstance(threads, int) or threads < 1:
        raise ProfileError("compute.threads must be a positive integer")
    return profile


def _keys(value):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key
            yield from _keys(inner)


def is_url(location):
    return isinstance(location, str) and "://" in location


def resolve(location, profile):
    """Absolute path or URL of a dataset location under the profile's data root."""
    if is_url(location) or Path(location).is_absolute():
        return location
    root = (profile or {}).get("data", {}).get("root")
    if not root:
        return str(Path(location).resolve())
    return f"{root.rstrip('/')}/{location}" if is_url(root) else str(Path(root, location))


def storage_options(profile):
    data = (profile or {}).get("data", {})
    options = dict(data.get("storage_options") or {})
    if data.get("credentials") == "anonymous":
        options.setdefault("anon", True)
    return options


def core_accepts_storage_options():
    import inspect

    from cellcanvas_core.data.data_set import DataSet

    return "storage_options" in inspect.signature(DataSet.from_paths).parameters


def dataset_arguments(paths, profile):
    """DataSet.from_paths arguments with locations resolved for this deployment.

    URL locations are opened by cellcanvas-core with the profile's storage options; that needs a core version
    whose DataSet.from_paths accepts ``storage_options``.
    """
    arguments = dict(paths)
    remote = False
    for key in READ_ONLY:
        if key not in arguments:
            continue
        value = arguments[key]
        resolved = [resolve(v, profile) for v in value] if isinstance(value, list) else resolve(value, profile)
        remote = remote or any(is_url(v) for v in (resolved if isinstance(resolved, list) else [resolved]))
        arguments[key] = resolved
    if remote:
        if not core_accepts_storage_options():
            raise ProfileError("URL data locations need cellcanvas-core with DataSet.from_paths(storage_options)")
        arguments["storage_options"] = storage_options(profile) or None
    return arguments


def check(profile_path, config_path=None, environ=None):
    profile = load_profile(profile_path, environ)
    report = {"profile": profile["name"], "bind": profile["bind"], "data_kind": profile["data"]["kind"],
              "state_dir": profile.get("state_dir"), "prompt_backends": profile["prompt_backends"],
              "providers": profile["providers"], "datasets": {}}
    config = config_path or profile.get("config")
    if config:
        datasets = json.loads(Path(config).read_text())["datasets"]
        for name, entry in datasets.items():
            report["datasets"][name] = {key: resolve(entry[key], profile) if isinstance(entry[key], str)
                                        else [resolve(v, profile) for v in entry[key]]
                                        for key in READ_ONLY if key in entry}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    checker = sub.add_parser("check", help="Validate a profile and resolve dataset locations")
    checker.add_argument("profile")
    checker.add_argument("config", nargs="?")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(check(args.profile, args.config), indent=2))
    except (ProfileError, OSError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
