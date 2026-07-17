import json
from typing import Any, Literal, cast

import dag_cbor
import pytest
import zarr
from dag_cbor.ipld import IPLDKind
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.internal.conjecture.data import ConjectureData
from testing_utils import CIDInMemoryCAS

from py_hamt import ShardedZarrStore

PROTOTYPE = zarr.core.buffer.default_buffer_prototype()
ARRAY_METADATA = json.dumps(
    {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4, 4],
        "data_type": "uint8",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [2, 2]},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": {},
    },
    separators=(",", ":"),
).encode()
CHUNKLESS_ARRAY_METADATA = json.dumps(
    {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4, 4],
    },
    separators=(",", ":"),
).encode()

CandidateFormat = Literal["v2", "v3", "dual"]
CandidateCondition = Literal[
    "valid",
    "undecodable",
    "non_dict",
    "wrong_shape",
    "missing_chunk_shape",
    "wrong_chunk_shape",
]
FailureKind = Literal["none", "coordinate", "real"]
CandidateSpec = tuple[str, CandidateFormat, CandidateCondition]
CandidateDocument = tuple[str, bytes, str, bool]

CANDIDATE_FORMATS: tuple[CandidateFormat, ...] = ("v2", "v3", "dual")
CANDIDATE_CONDITIONS: tuple[CandidateCondition, ...] = (
    "valid",
    "undecodable",
    "non_dict",
    "wrong_shape",
    "missing_chunk_shape",
    "wrong_chunk_shape",
)
COORDINATE_NAMES = (
    "lat",
    "lon",
    "time",
    "latitude",
    "longitude",
    "forecast_reference_time",
    "step",
)
COORDINATE_PATHS = ("lat", "/lon", "group/time", *COORDINATE_NAMES[3:])


def buf(data: bytes) -> zarr.core.buffer.Buffer:
    return PROTOTYPE.buffer.from_bytes(data)


def decode_root(data: bytes) -> dict[str, Any]:
    root_obj = dag_cbor.decode(data)
    assert isinstance(root_obj, dict)
    return cast(dict[str, Any], root_obj)


def _candidate_payload(
    path: str,
    metadata_format: Literal["v2", "v3"],
    condition: CandidateCondition,
) -> bytes:
    if condition == "undecodable":
        return b"\xffnot-json:" + path.encode()
    if condition == "non_dict":
        return json.dumps([path, metadata_format]).encode()

    metadata: dict[str, object] = {
        "candidate": path,
        "zarr_format": 2 if metadata_format == "v2" else 3,
        "shape": [8, 8] if condition == "wrong_shape" else [4, 4],
    }
    if condition != "missing_chunk_shape":
        declared_chunks = [4, 4] if condition == "wrong_chunk_shape" else [2, 2]
        if metadata_format == "v2":
            metadata["chunks"] = declared_chunks
        else:
            metadata["chunk_grid"] = {
                "name": "regular",
                "configuration": {"chunk_shape": declared_chunks},
            }
    return json.dumps(metadata, separators=(",", ":")).encode()


def _candidate_documents(spec: CandidateSpec) -> list[CandidateDocument]:
    path, metadata_format, condition = spec
    normalized_path = path.strip("/")
    coordinate = normalized_path.rsplit("/", 1)[-1] in COORDINATE_NAMES
    formats: tuple[Literal["v2", "v3"], ...] = (
        ("v2", "v3") if metadata_format == "dual" else (metadata_format,)
    )
    return [
        (
            f"{path}/.zarray" if document_format == "v2" else f"{path}/zarr.json",
            _candidate_payload(path, document_format, condition),
            normalized_path,
            coordinate,
        )
        for document_format in formats
    ]


def _pinned_data() -> st.DataObject:
    """Return deterministic draws for explicit examples using ``st.data()``."""
    return st.DataObject(ConjectureData.for_choices([0] * 8))


@pytest.mark.asyncio
async def test_v1_legacy_root_infers_primary_array_path_for_listing() -> None:
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    await store.set("temp/zarr.json", buf(ARRAY_METADATA))
    await store.set("temp/c/0/0", buf(b"chunk"))
    root = await store.flush()

    root_obj = decode_root(await cas.load(root))
    root_obj["chunks"].pop("primary_array_path")
    legacy_root = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=str(legacy_root),
    )

    chunk = await reopened.get("temp/c/0/0", PROTOTYPE)
    assert chunk is not None
    assert chunk.to_bytes() == b"chunk"

    listed = [key async for key in reopened.list()]
    assert "temp/c/0/0" in listed and "c/0/0" not in listed
    assert [key async for key in reopened.list_prefix("temp/c/")] == ["temp/c/0/0"]


@pytest.mark.asyncio
async def test_inference_rejects_lone_chunkless_candidate() -> None:
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    await store.set("temp/zarr.json", buf(ARRAY_METADATA))
    await store.set("temp/c/0/0", buf(b"chunk"))
    root = await store.flush()

    root_obj = decode_root(await cas.load(root))
    root_obj["chunks"].pop("primary_array_path")
    malformed_cid = await cas.save(CHUNKLESS_ARRAY_METADATA, codec="raw")
    root_obj["metadata"] = {"malformed/zarr.json": malformed_cid}
    legacy_root = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=str(legacy_root),
    )
    assert reopened._primary_array_path in (None, "")


@pytest.mark.asyncio
async def test_inference_ignores_chunkless_candidate_beside_valid_array() -> None:
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    await store.set("temp/zarr.json", buf(ARRAY_METADATA))
    await store.set("temp/c/0/0", buf(b"chunk"))
    root = await store.flush()

    root_obj = decode_root(await cas.load(root))
    root_obj["chunks"].pop("primary_array_path")
    root_obj["metadata"]["malformed/zarr.json"] = await cas.save(
        CHUNKLESS_ARRAY_METADATA,
        codec="raw",
    )
    legacy_root = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=str(legacy_root),
    )
    assert reopened._primary_array_path == "temp"


class FlakyMetadataCAS(CIDInMemoryCAS):
    """Fails loads of one specific CID to simulate a partial CAS outage."""

    def __init__(self) -> None:
        super().__init__()
        self.failing_cid: object = None

    async def load(
        self,
        identifier: IPLDKind,
        offset: int | None = None,
        length: int | None = None,
        suffix: int | None = None,
    ) -> bytes:
        if self.failing_cid is not None and str(identifier) == str(self.failing_cid):
            raise ConnectionError("simulated partial CAS outage")
        return await super().load(
            identifier,
            offset=offset,
            length=length,
            suffix=suffix,
        )


@pytest.mark.asyncio
async def test_inference_aborts_when_a_candidate_cannot_be_loaded() -> None:
    """If any candidate's metadata cannot be loaded, the true primary might be
    the unreadable one — inference must abort (keep the legacy '' default)
    rather than confidently adopt a surviving same-shape candidate."""
    cas = FlakyMetadataCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    await store.set("temp/zarr.json", buf(ARRAY_METADATA))
    await store.set("precip/zarr.json", buf(ARRAY_METADATA))
    await store.set("temp/c/0/0", buf(b"chunk"))
    root = await store.flush()

    root_obj = decode_root(await cas.load(root))
    root_obj["chunks"].pop("primary_array_path")
    legacy_root = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")

    # Make temp's metadata unreadable: only "precip" would survive the scan.
    metadata = root_obj["metadata"]
    cas.failing_cid = metadata["temp/zarr.json"]

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=str(legacy_root),
    )
    assert reopened._primary_array_path in (None, ""), (
        "a partial CAS outage must not rebind shard data under a surviving "
        f"candidate, got {reopened._primary_array_path!r}"
    )


@pytest.mark.asyncio
async def test_inference_skips_non_candidates_and_dedupes_dual_format() -> None:
    """The scan must skip coordinate arrays, undecodable blobs, shape and
    chunk-shape mismatches, non-string keys, and count an array once even
    when it registers both zarr.json and v2 .zarray metadata."""
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    await store.set("temp/zarr.json", buf(ARRAY_METADATA))
    await store.set("temp/c/0/0", buf(b"chunk"))
    root = await store.flush()

    def meta(payload: dict[str, object]) -> bytes:
        return json.dumps(payload).encode()

    root_obj = decode_root(await cas.load(root))
    root_obj["chunks"].pop("primary_array_path")
    metadata = root_obj["metadata"]
    metadata["lat/zarr.json"] = await cas.save(
        meta({
            "shape": [4, 4],
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [2, 2]},
            },
        }),
        codec="raw",
    )
    metadata["garbage/zarr.json"] = await cas.save(b"not-json", codec="raw")
    metadata["wrongshape/zarr.json"] = await cas.save(
        meta({"shape": [8, 8]}), codec="raw"
    )
    metadata["wrongchunks/zarr.json"] = await cas.save(
        meta({
            "shape": [4, 4],
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [4, 4]},
            },
        }),
        codec="raw",
    )
    # v2-format dual registration of the same array: "chunks" key, same grid.
    metadata["temp/.zarray"] = await cas.save(
        meta({"shape": [4, 4], "chunks": [2, 2]}), codec="raw"
    )
    legacy_root = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=str(legacy_root),
    )
    assert reopened._primary_array_path == "temp"
    listed = [key async for key in reopened.list()]
    assert "temp/c/0/0" in listed


@pytest.mark.asyncio
async def test_inference_handles_malformed_metadata_maps() -> None:
    """Direct-call coverage for defensive branches dag-cbor cannot produce:
    a non-mapping metadata value and a non-string metadata key."""
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    store._root_obj["chunks"].pop("primary_array_path", None)

    store._root_obj["metadata"] = None
    await store._infer_v1_legacy_primary_array_path()
    assert store._primary_array_path in (None, "")

    cid = await cas.save(b"payload", codec="raw")
    store._root_obj["metadata"] = {1: cid}
    await store._infer_v1_legacy_primary_array_path()
    assert store._primary_array_path in (None, "")


@pytest.mark.asyncio
@example(
    data=_pinned_data(),
    first_path="temp",
    first_format="v3",
    first_condition="valid",
    second_path="precip",
    second_format="v2",
    second_condition="wrong_shape",
    coordinate_path="lat",
    coordinate_format="v3",
    coordinate_condition="valid",
    failure_kind="none",
)
@example(
    data=_pinned_data(),
    first_path="temp",
    first_format="v2",
    first_condition="valid",
    second_path="precip",
    second_format="v3",
    second_condition="wrong_chunk_shape",
    coordinate_path="/lon",
    coordinate_format="v2",
    coordinate_condition="valid",
    failure_kind="coordinate",
)
@given(
    data=st.data(),
    first_path=st.sampled_from(("temp", "/temp", "group/temp")),
    first_format=st.sampled_from(CANDIDATE_FORMATS),
    first_condition=st.sampled_from(CANDIDATE_CONDITIONS),
    second_path=st.sampled_from(("precip", "/precip", "group/precip")),
    second_format=st.sampled_from(CANDIDATE_FORMATS),
    second_condition=st.sampled_from(CANDIDATE_CONDITIONS),
    coordinate_path=st.sampled_from(COORDINATE_PATHS),
    coordinate_format=st.sampled_from(CANDIDATE_FORMATS),
    coordinate_condition=st.sampled_from(CANDIDATE_CONDITIONS),
    failure_kind=st.sampled_from(("none", "coordinate", "real")),
)
@settings(max_examples=25, deadline=None)
async def test_v1_legacy_primary_inference_property(
    data: st.DataObject,
    first_path: str,
    first_format: CandidateFormat,
    first_condition: CandidateCondition,
    second_path: str,
    second_format: CandidateFormat,
    second_condition: CandidateCondition,
    coordinate_path: str,
    coordinate_format: CandidateFormat,
    coordinate_condition: CandidateCondition,
    failure_kind: FailureKind,
) -> None:
    """Inference is order-independent, conservative, and never persisted."""
    cas = FlakyMetadataCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    await store.set("temp/c/0/0", buf(b"chunk"))
    root_cid = await store.flush()
    root_obj = decode_root(await cas.load(root_cid))
    root_obj["chunks"].pop("primary_array_path")

    specs: tuple[CandidateSpec, ...] = (
        (first_path, first_format, first_condition),
        (second_path, second_format, second_condition),
        (coordinate_path, coordinate_format, coordinate_condition),
    )
    documents = [document for spec in specs for document in _candidate_documents(spec)]
    ordered_documents = data.draw(
        st.permutations(documents),
        label="metadata_insertion_order",
    )

    metadata: dict[str, IPLDKind] = {}
    saved_documents: list[tuple[CandidateDocument, IPLDKind]] = []
    for document in ordered_documents:
        key, payload, _, _ = document
        cid = await cas.save(payload, codec="raw")
        metadata[key] = cid
        saved_documents.append((document, cid))
    root_obj["metadata"] = metadata

    failing_cid: IPLDKind | None = None
    if failure_kind != "none":
        eligible_documents = [
            saved
            for saved in saved_documents
            if saved[0][3] is (failure_kind == "coordinate")
        ]
        _, failing_cid = data.draw(
            st.sampled_from(eligible_documents),
            label=f"{failure_kind}_load_failure_position",
        )

    original_chunks = dict(root_obj["chunks"])
    legacy_root = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")
    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        root_cid=str(legacy_root),
    )
    # DAG-CBOR canonicalizes map ordering. Restore the generated order for the
    # direct inference seam used by the deterministic defensive tests above.
    reopened._root_obj["metadata"] = metadata
    reopened._primary_array_path = ""
    cas.failing_cid = failing_cid
    await reopened._infer_v1_legacy_primary_array_path()

    valid_paths = {
        path.strip("/")
        for path, _, condition in specs
        if condition == "valid"
        and path.strip("/").rsplit("/", 1)[-1] not in COORDINATE_NAMES
    }
    expected_path = (
        next(iter(valid_paths))
        if failure_kind != "real" and len(valid_paths) == 1
        else ""
    )
    assert reopened._primary_array_path == expected_path
    assert reopened._root_obj["chunks"] == original_chunks

    flushed_root = await reopened.flush()
    assert flushed_root == str(legacy_root)
    assert decode_root(await cas.load(flushed_root))["chunks"] == original_chunks
