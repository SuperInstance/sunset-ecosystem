"""Tests for fleet.deck — Presentation deck generation."""

from fleet.deck import Slide, Deck, breeding_report, fleet_status, flux_gate_decision, architecture_decision, research_summary


class TestSlide:
    def test_slide_creation(self):
        s = Slide("Title", ["a", "b"], "quote")
        assert s.title == "Title"
        assert s.bullets == ["a", "b"]
        assert s.quote == "quote"

    def test_slide_optional_quote(self):
        s = Slide("Title", ["a"])
        assert s.quote is None


class TestDeck:
    def test_add_slide(self):
        d = Deck("Report", "breeding")
        d.add(Slide("S1", ["b1"]))
        assert len(d.slides) == 1

    def test_render_markdown(self):
        d = Deck("Report", "breeding")
        d.add(Slide("Overview", ["a", "b"], "Q"))
        md = d.render()
        assert "# Report" in md
        assert "Overview" in md
        assert "a" in md
        assert "Q" in md

    def test_to_dict(self):
        d = Deck("Report", "breeding")
        d.add(Slide("S1", ["b1"]))
        doc = d.to_dict()
        assert doc["title"] == "Report"
        assert doc["deck_type"] == "breeding"
        assert len(doc["slides"]) == 1

    def test_breeding_report(self):
        md = breeding_report(
            generation=42,
            pool_size=50,
            pass_rate=0.85,
            top_score=0.12,
            flux_gate_blocks=3,
            thermal_violations=0,
            proof_count=47,
        )
        assert "42" in md
        assert "50" in md
        assert "85" in md or "0.85" in md
        assert "3" in md
        assert "47" in md

    def test_fleet_status(self):
        md = fleet_status(
            services_up=10,
            services_down=2,
            breeding_active=True,
            last_proof="abc123",
            blockers=["thermal"],
        )
        assert "10" in md
        assert "2" in md
        assert "abc123" in md
        assert "thermal" in md

    def test_flux_gate_decision(self):
        md = flux_gate_decision(
            candidate_id="c1",
            passed=True,
            score=0.9,
            violations={"bound": 0.1},
            proof_hash="abc123",
            vm_cycles=42,
        )
        assert "c1" in md
        assert "PASS" in md
        assert "0.9" in md
        assert "abc123" in md

    def test_architecture_decision(self):
        md = architecture_decision(
            title="T",
            problem="P",
            options="O",
            recommendation="R",
            risk="Low",
            timeline="1w",
        )
        assert "T" in md
        assert "P" in md
        assert "R" in md

    def test_research_summary(self):
        md = research_summary(
            title="T",
            what_learned="W",
            why_matters="M",
            what_to_do="D",
        )
        assert "T" in md
        assert "W" in md
        assert "M" in md
        assert "D" in md
