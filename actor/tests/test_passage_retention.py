import asyncio
import json

from test_read_batch_capacity import make_session

from cayu_tau3_banking.product_navigation import COMPARISON_TOOL, READ_TOOL, ProductNavigation


def test_duplicate_save_and_document_id_recovery_are_bounded_and_source_checked(tmp_path):
    session = make_session(tmp_path, count=2, content="First relevant term.\n\nSecond condition.")
    session.advisory_workflow = True
    async def run():
        entry = "doc_savings_accounts_nimbus_000"
        unread = "doc_savings_accounts_nimbus_001"
        read = await session.invoke(READ_TOOL, {"documents": [{"entry_id": entry}]})
        assert not read.is_error, read.content
        body = json.loads(read.content)
        handles = list(body["documents"][0]["passages"])
        saved = json.loads((await session.invoke(COMPARISON_TOOL, {"passage_handles": handles + handles})).content)
        assert saved["accepted_handles"] == handles
        recovered = json.loads((await session.invoke(COMPARISON_TOOL, {"operation": "read", "entry_ids": [entry, entry, unread]})).content)
        assert [p["passage_handle"] for p in recovered["pinned_passages"]] == handles
        assert recovered["errors"] == [{"entry_id": unread, "error": "No retained passages for this source in this worksheet."}]
        assert not session.controller.has_delivered_knowledge_revision(unread, "1")
        # Recovery never becomes an alternate way to pin unread documents.
        for args in [{"entry_ids": [unread], "operation": "append"},
                     {"entry_ids": [entry], "passage_handles": [], "operation": "read"}]:
            assert (await session.invoke(COMPARISON_TOOL, args)).is_error
    asyncio.run(run())


def test_document_recovery_cursor_is_bound_to_selection_and_saved_state(tmp_path):
    session = make_session(tmp_path, count=3, content="\n\n".join(str(i) + "x" * 1400 for i in range(5)))
    session.advisory_workflow = True
    async def invoke(**args):
        result = await session.invoke(COMPARISON_TOOL, args)
        return result, json.loads(result.content) if not result.is_error else None
    async def run():
        entries = [f"doc_savings_accounts_nimbus_{i:03d}" for i in range(3)]
        read = await session.invoke(READ_TOOL, {"documents": [{"entry_id": entry} for entry in entries]})
        assert not read.is_error, read.content
        body = json.loads(read.content)
        handles = [h for doc in body["documents"] for h in doc["passages"]]
        await invoke(passage_handles=handles)
        _, first = await invoke(operation="read", entry_ids=entries)
        assert first["next_cursor"] and len(json.dumps(first["pinned_passages"]).encode()) < 12000
        recovered = list(first["pinned_passages"])
        cursor = first["next_cursor"]
        while cursor:
            _, page = await invoke(operation="read", entry_ids=entries, cursor=cursor)
            recovered.extend(page["pinned_passages"])
            cursor = page["next_cursor"]
        assert [p["passage_handle"] for p in recovered] == handles
        changed, _ = await invoke(operation="read", entry_ids=entries[:1], cursor=first["next_cursor"])
        assert changed.is_error and "Stale" in changed.content
        await invoke(operation="replace", passage_handles=handles[:1])
        stale, _ = await invoke(operation="read", entry_ids=entries, cursor=first["next_cursor"])
        assert stale.is_error and "Stale" in stale.content
    asyncio.run(run())


def test_pin_conflicting_passages_without_scalar_extraction_and_restore(tmp_path):
    content = "Annual yield is 4.2%. Conditions apply.\n\nExample annual yield is 4.8%. Conflicting source assertion."
    session = make_session(tmp_path, count=1, content=content)
    session.advisory_workflow = True
    args = {"documents": [{"entry_id": "doc_savings_accounts_nimbus_000"}]}
    result = asyncio.run(session.invoke(READ_TOOL, args))
    body = json.loads(result.content)
    assert body["read_progress"]["newly_delivered"] == ["doc_savings_accounts_nimbus_000"]
    handles = list(body["documents"][0]["passages"])
    assert len(handles) == 2
    pinned = asyncio.run(session.invoke(COMPARISON_TOOL, {"passage_handles": handles}))
    assert not pinned.is_error, pinned.content
    assert len(json.loads(pinned.content)["accepted_handles"]) == 2
    session.navigation = ProductNavigation(session.corpus, session.navigation.snapshot())
    rendered = asyncio.run(session.runtime_context())
    assert "Annual yield is 4.2%. Conditions apply." in rendered
    assert "Example annual yield is 4.8%. Conflicting source assertion." in rendered
    repeated = json.loads(asyncio.run(session.invoke(READ_TOOL, args)).content)
    assert repeated["read_progress"]["newly_delivered"] == []
    assert repeated["read_progress"]["all_sources_previously_delivered"]


def test_passage_handle_does_not_authorize_unread_source_or_invented_handle(tmp_path):
    session = make_session(tmp_path, count=1)
    session.advisory_workflow = True
    handle = session.navigation.document_passages["doc_savings_accounts_nimbus_000"][0][
        "passage_handle"
    ]
    result = asyncio.run(session.invoke(COMPARISON_TOOL, {"passage_handles": [handle]}))
    assert not json.loads(result.content)["accepted_handles"]
    assert json.loads(result.content)["errors"]
    assert session.navigation.snapshot()["worksheet"] is None
    before = session.navigation.snapshot()
    result = asyncio.run(session.invoke(COMPARISON_TOOL, {"passage_handles": ["pass_invented"]}))
    assert json.loads(result.content)["errors"]
    assert session.navigation.snapshot()["worksheet"] == before["worksheet"]


def test_large_passage_locators_are_bounded_and_do_not_change_source_text(tmp_path):
    content = ("Long source paragraph with repeated words and conditions. " * 75).strip()
    session = make_session(tmp_path, count=1, content=content)
    result = asyncio.run(
        session.invoke(READ_TOOL, {"documents": [{"entry_id": "doc_savings_accounts_nimbus_000"}]})
    )
    body = json.loads(result.content)["documents"][0]
    assert body["text"] == content
    assert len(body["passages"]) > 1
    for start, end in body["passages"].values():
        assert 8 <= end - start <= 1600


def test_public_api_empty_recovery_and_short_or_unbroken_passages(tmp_path):
    content = "Yes.\n\n" + "z" * 1701
    session = make_session(tmp_path, count=1, content=content)
    session.advisory_workflow = True
    schema = next(s for s in session.tool_schemas() if s["name"] == COMPARISON_TOOL)
    assert set(schema["input_schema"]["properties"]) == {
        "passage_handles",
        "worksheet_id",
        "operation",
        "cursor",
        "entry_ids",
    }
    body = json.loads(
        asyncio.run(
            session.invoke(
                READ_TOOL, {"documents": [{"entry_id": "doc_savings_accounts_nimbus_000"}]}
            )
        ).content
    )
    handles = list(body["documents"][0]["passages"])
    result = asyncio.run(session.invoke(COMPARISON_TOOL, {"passage_handles": handles}))
    assert not result.is_error, result.content
    pinned = json.loads(result.content)
    assert not pinned["errors"], pinned
    assert len(pinned["accepted_handles"]) == len(handles)
    recovered = json.loads(
        asyncio.run(session.invoke(COMPARISON_TOOL, {"passage_handles": []})).content
    )
    assert [p["passage_handle"] for p in recovered["pinned_passages"]] == pinned["accepted_handles"]
    assert recovered["authorizes_completion"] is False


def test_window_read_does_not_authorize_pin(tmp_path):
    session = make_session(tmp_path, count=1, content="A policy condition. " * 30)
    session.advisory_workflow = True
    result = asyncio.run(
        session.invoke(
            READ_TOOL,
            {
                "documents": [{"entry_id": "doc_savings_accounts_nimbus_000"}],
                "source_span": {"start": 0, "end": 30},
            },
        )
    )
    assert not result.is_error, result.content
    body = json.loads(result.content)
    assert body["read_progress"]["newly_delivered"] == []
    handle = session.navigation.document_passages["doc_savings_accounts_nimbus_000"][0][
        "passage_handle"
    ]
    pin = json.loads(
        asyncio.run(session.invoke(COMPARISON_TOOL, {"passage_handles": [handle]})).content
    )
    assert not pin["accepted_handles"]
    assert pin["errors"]


def test_retention_capacity_partial_acceptance_atomic_replace_and_bounded_recovery(tmp_path):
    async def scenario():
        session = make_session(
            tmp_path,
            count=12,
            content="\n\n".join(
                f"Condition {i}: " + "A material condition requires verification. " * 25
                for i in range(10)
            ),
        )
        session.advisory_workflow = True
        entries = list(session.navigation.document_passages)
        for start in range(0, len(entries), 4):
            result = await session.invoke(
                READ_TOOL, {"documents": [{"entry_id": e} for e in entries[start : start + 4]]}
            )
            assert not result.is_error, result.content
        handles = [
            p["passage_handle"]
            for rows in session.navigation.document_passages.values()
            for p in rows
        ]

        async def invoke(**args):
            result = await session.invoke(COMPARISON_TOOL, args)
            assert not result.is_error, result.content
            assert len(result.content.encode()) < 20_000
            return json.loads(result.content)

        mixed = await invoke(passage_handles=[handles[0], "typo"])
        assert mixed["accepted_handles"] == [handles[0]]
        assert len(mixed["errors"]) == 1
        for start in range(0, 96, 24):
            saved = await invoke(passage_handles=handles[start : start + 24])
            assert "pinned_passages" not in saved
            assert not saved["errors"]
        full = await invoke(passage_handles=handles[96:])
        assert full["retained_count"] == 96
        assert not full["accepted_handles"]
        assert all("capacity" in e["error"] for e in full["errors"])
        # A later retained passage can be fetched without echoing the earlier 95.
        selected = await invoke(passage_handles=[handles[95]], operation="read")
        assert [p["passage_handle"] for p in selected["pinned_passages"]] == [handles[95]]
        seen = []
        cursor = None
        while True:
            page = await invoke(
                passage_handles=[], operation="read", **({"cursor": cursor} if cursor else {})
            )
            seen.extend(p["passage_handle"] for p in page["pinned_passages"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
        assert seen == handles[:96]
        before = session.navigation.snapshot()["worksheet"]
        rejected = await invoke(passage_handles=[handles[-1], "typo"], operation="replace")
        assert not rejected["replacement_applied"]
        assert session.navigation.snapshot()["worksheet"] == before
        replacement = await invoke(passage_handles=[handles[-1]], operation="replace")
        assert replacement["replacement_applied"] and replacement["retained_count"] == 1
        restored = ProductNavigation(session.corpus, session.navigation.snapshot())
        assert [f["field"] for f in restored.snapshot()["worksheet"]["arguments"]["facts"]] == [
            handles[-1]
        ]
        cleared = await invoke(passage_handles=[], operation="replace")
        assert cleared["retained_count"] == 0
        assert session.navigation.snapshot()["worksheet"] is None

    asyncio.run(scenario())


def test_product_capacity_preserves_fitting_append_and_rejects_replacement(tmp_path):
    async def scenario():
        for i in range(17):
            entry = f"doc_savings_accounts_p{i:02d}_001"
            (tmp_path / f"{entry}.json").write_text(
                json.dumps(
                    {
                        "id": entry,
                        "title": f"Product {i}",
                        "content": "First source condition.\n\nSecond source condition.",
                    }
                )
            )
        session = make_session(tmp_path, count=0)
        session.advisory_workflow = True
        nav = session.navigation
        assert len(nav.products) == 17
        entries = list(nav.document_passages)
        for batch in (entries[:16], entries[16:]):
            result = await session.invoke(
                READ_TOOL, {"documents": [{"entry_id": e} for e in batch]}
            )
            assert not result.is_error, result.content
        groups = [nav.document_passages[e] for e in entries]
        first = [g[0]["passage_handle"] for g in groups]

        async def invoke(handles, operation="append"):
            result = await session.invoke(
                COMPARISON_TOOL, {"passage_handles": handles, "operation": operation}
            )
            assert not result.is_error, result.content
            return json.loads(result.content)

        assert (await invoke(first[:16]))["retained_count"] == 16
        fitting = groups[0][1]["passage_handle"]
        result = await invoke([first[16], fitting])
        assert result["accepted_handles"] == [fitting]
        assert "16-product" in result["errors"][0]["error"]
        assert result["retained_product_count"] == result["product_capacity"] == 16
        before = nav.snapshot()["worksheet"]
        result = await invoke(first, "replace")
        assert not result["replacement_applied"] and not result["accepted_handles"]
        assert "16-product" in result["errors"][0]["error"]
        assert nav.snapshot()["worksheet"] == before

    asyncio.run(scenario())


def test_recovery_cursor_detects_mutations_and_selection_changes(tmp_path):
    async def scenario():
        session = make_session(tmp_path, count=16, content="Condition. " * 130)
        session.advisory_workflow = True
        nav = session.navigation
        entries = list(nav.document_passages)
        for start in range(0, 16, 4):
            result = await session.invoke(
                READ_TOOL, {"documents": [{"entry_id": e} for e in entries[start : start + 4]]}
            )
            assert not result.is_error, result.content
        handles = [nav.document_passages[e][0]["passage_handle"] for e in entries]

        async def invoke(**args):
            return await session.invoke(COMPARISON_TOOL, args)

        async def page():
            return json.loads((await invoke(passage_handles=[], operation="read")).content)

        await invoke(passage_handles=handles[:15])
        cursor = (await page())["next_cursor"]
        assert cursor
        # An identical re-save and restoring durable state preserve the cursor.
        await invoke(passage_handles=[handles[0]])
        session.navigation = ProductNavigation(session.corpus, session.navigation.snapshot())
        continued = await invoke(passage_handles=[], operation="read", cursor=cursor)
        assert not continued.is_error, continued.content
        wrong_selection = await invoke(
            passage_handles=[handles[0]], operation="read", cursor=cursor
        )
        assert wrong_selection.is_error and "cursor" in wrong_selection.content
        await invoke(passage_handles=[handles[15]])
        stale = await invoke(passage_handles=[], operation="read", cursor=cursor)
        assert stale.is_error and "Stale" in stale.content
        cursor = (await page())["next_cursor"]
        await invoke(passage_handles=[handles[-1]], operation="replace")
        stale = await invoke(passage_handles=[], operation="read", cursor=cursor)
        assert stale.is_error and "Stale" in stale.content
        restarted = await page()
        assert [p["passage_handle"] for p in restarted["pinned_passages"]] == [handles[-1]]
        assert restarted["next_cursor"] is None
        malformed = await invoke(passage_handles=[], operation="read", cursor="bad")
        assert malformed.is_error

    asyncio.run(scenario())
