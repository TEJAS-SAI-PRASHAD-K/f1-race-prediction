"""API tests against a real Postgres test database."""

from __future__ import annotations


class TestHealth:
    def test_health_checks_the_database(self, client):
        """A health check that does not touch the DB is close to useless."""
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["database"] == "connected"
        assert body["laps_ingested"] is not None


class TestCatalog:
    def test_lists_seasons(self, client, seeded):
        assert client.get("/api/v1/seasons").json() == [{"year": 2024}]

    def test_lists_events(self, client, seeded):
        events = client.get("/api/v1/seasons/2024/events").json()
        assert len(events) == 1
        assert events[0]["event_name"] == "Italian Grand Prix"
        assert events[0]["round"] == 16

    def test_unknown_season_is_404(self, client, seeded):
        assert client.get("/api/v1/seasons/1999/events").status_code == 404

    def test_lists_sessions_for_event(self, client, seeded):
        sessions = client.get(f"/api/v1/events/{seeded['event_id']}/sessions").json()
        assert [s["session_type"] for s in sessions] == ["R"]
        assert sessions[0]["total_laps"] == 53

    def test_lists_drivers(self, client, seeded):
        abbrs = {d["abbreviation"] for d in client.get("/api/v1/drivers").json()}
        assert abbrs == {"VER", "LEC"}


class TestLaps:
    def test_returns_laps_with_derived_fields(self, client, seeded):
        laps = client.get(f"/api/v1/sessions/{seeded['session_id']}/laps").json()
        assert len(laps) == 20
        lap = laps[0]
        # Milliseconds are the stored truth; seconds are a convenience.
        assert lap["lap_time_ms"] == 84_100
        assert lap["lap_time_s"] == 84.1
        assert lap["track_status"] == [1]
        assert lap["is_green_flag"] is True

    def test_track_status_round_trips_as_an_array(self, client, seeded):
        """The array must survive Postgres and JSON serialisation intact."""
        laps = client.get(f"/api/v1/sessions/{seeded['session_id']}/laps").json()
        sc_lap = next(lap for lap in laps if lap["lap_number"] == 7)
        assert sc_lap["track_status"] == [1, 4]
        assert sc_lap["is_green_flag"] is False

    def test_filter_by_driver(self, client, seeded):
        laps = client.get(
            f"/api/v1/sessions/{seeded['session_id']}/laps", params={"driver": "ver"}
        ).json()
        assert len(laps) == 10
        assert {lap["driver"] for lap in laps} == {"VER"}

    def test_filter_by_compound(self, client, seeded):
        laps = client.get(
            f"/api/v1/sessions/{seeded['session_id']}/laps", params={"compound": "hard"}
        ).json()
        assert {lap["compound"] for lap in laps} == {"HARD"}

    def test_green_only_excludes_safety_car_laps(self, client, seeded):
        laps = client.get(
            f"/api/v1/sessions/{seeded['session_id']}/laps", params={"green_only": True}
        ).json()
        assert 7 not in {lap["lap_number"] for lap in laps}
        assert all(lap["is_green_flag"] for lap in laps)

    def test_unknown_session_is_404(self, client, seeded):
        assert client.get("/api/v1/sessions/999999/laps").status_code == 404


class TestResults:
    def test_results_ordered_by_position(self, client, seeded):
        results = client.get(f"/api/v1/sessions/{seeded['session_id']}/results").json()
        assert [r["driver"] for r in results] == ["LEC", "VER"]
        assert results[0]["position"] == 1.0
        assert results[0]["grid_position"] == 4.0


class TestStints:
    def test_aggregates_laps_into_stints(self, client, seeded):
        stints = client.get(f"/api/v1/sessions/{seeded['session_id']}/stints").json()
        assert len(stints) == 4  # 2 drivers x 2 stints

        first = next(s for s in stints if s["driver"] == "LEC" and s["stint_number"] == 1)
        assert first["compound"] == "MEDIUM"
        assert (first["lap_start"], first["lap_end"]) == (1, 5)
        assert first["laps_run"] == 5
        assert first["green_laps"] == 5

    def test_median_pace_excludes_the_safety_car_lap(self, client, seeded):
        """Lap 7 is 20s slower and under safety car. Including it would make
        the stint look far slower than the car actually was -- exactly the
        contamination Phase 2 must avoid."""
        stints = client.get(f"/api/v1/sessions/{seeded['session_id']}/stints").json()
        second = next(s for s in stints if s["driver"] == "VER" and s["stint_number"] == 2)

        assert second["laps_run"] == 5, "the SC lap is still part of the stint"
        assert second["green_laps"] == 4, "but it does not count toward pace"
        # Green laps 6,8,9,10 -> tyre_life 1,3,4,5 -> 84100,84300,84400,84500
        assert second["median_green_lap_time_ms"] == 84_350
        assert second["median_green_lap_time_ms"] < 90_000


class TestCors:
    """CORS is what lets the separate frontend read a response at all.

    Without `Access-Control-Allow-Origin` the request still reaches the API and
    still returns 200 -- the browser simply refuses to hand the body to the
    page, which is why this is easy to miss in curl and impossible to miss in a
    browser.
    """

    def test_allows_the_configured_dev_origin(self, client):
        response = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert response.status_code == 200
        assert (
            response.headers["access-control-allow-origin"] == "http://localhost:5173"
        )

    def test_allows_the_loopback_spelling_too(self, client):
        """A browser treats localhost and 127.0.0.1 as different origins, so
        allowing one does not allow the other."""
        response = client.get("/health", headers={"Origin": "http://127.0.0.1:5173"})
        assert (
            response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
        )

    def test_an_unconfigured_origin_gets_no_allow_header(self, client):
        """The point of not defaulting to "*": an origin nobody configured is
        not echoed back, so the browser blocks the read."""
        response = client.get("/health", headers={"Origin": "https://example.invalid"})
        assert "access-control-allow-origin" not in response.headers

    def test_preflight_is_answered(self, client):
        """A JSON request with a Content-Type header triggers a preflight; if
        OPTIONS is not answered the real request is never sent."""
        response = client.options(
            "/api/v1/seasons",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert response.status_code == 200
        assert (
            response.headers["access-control-allow-origin"] == "http://localhost:5173"
        )
        assert "GET" in response.headers["access-control-allow-methods"]


class TestCorsSettings:
    def test_origins_are_split_on_commas(self):
        from app.config import Settings

        settings = Settings(cors_origins="https://a.example, https://b.example")
        assert settings.cors_origin_list == ["https://a.example", "https://b.example"]

    def test_blank_entries_are_dropped(self):
        """A trailing comma in a .env file would otherwise become an empty
        origin, which matches nothing and silently looks like a typo."""
        from app.config import Settings

        assert Settings(cors_origins="https://a.example,,").cors_origin_list == [
            "https://a.example"
        ]

    def test_the_default_is_not_a_wildcard(self):
        """Checks the field default rather than a loaded Settings, so the
        assertion is about the committed code and not about whoever's .env
        happens to be on disk."""
        from app.config import Settings

        default = Settings.model_fields["cors_origins"].default
        assert "*" not in default
        assert default == "http://localhost:5173,http://127.0.0.1:5173"
