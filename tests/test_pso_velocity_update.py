import numpy as np

from algorithms.pso import PSO
from tests.fixtures.small_mazes import trivial_3x3


def _make_pso():
    return PSO(trivial_3x3(), num_particles=1, omega=1.0, c1=0.1, c2=0.2)


def _make_particle(position, velocity, personal_best):
    return {
        "position": np.array(position, dtype=float),
        "velocity": np.array(velocity, dtype=float),
        "personal_best": np.array(personal_best, dtype=float),
    }


def test_velocity_update_matches_canonical_formula(monkeypatch):
    """Per Shi & Eberhart 1998 Eq. 1 (inertia weight form): v(t+1) = ω*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)"""
    pso = _make_pso()
    particle = _make_particle((2.0, 3.0), (0.5, 0.1), (4.0, 5.0))
    gbest = np.array((6.0, 7.0), dtype=float)

    draws = iter([
        np.array((0.5, 0.5), dtype=float),
        np.array((0.5, 0.5), dtype=float),
    ])

    monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: next(draws))

    updated = pso.update_velocity(particle, gbest)

    np.testing.assert_allclose(updated["velocity"], np.array((1.0, 0.6), dtype=float))


def test_velocity_draws_fresh_r1_r2_per_call(monkeypatch):
    """Per Shi & Eberhart 1998: r1, r2 are drawn fresh each iteration per dimension"""
    pso = _make_pso()
    gbest = np.array((6.0, 7.0), dtype=float)

    calls = iter([
        np.array((0.1, 0.9), dtype=float),
        np.array((0.2, 0.8), dtype=float),
        np.array((0.9, 0.1), dtype=float),
        np.array((0.8, 0.2), dtype=float),
    ])

    monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: next(calls))

    first = pso.update_velocity(_make_particle((2.0, 3.0), (0.5, 0.1), (4.0, 5.0)), gbest)
    second = pso.update_velocity(_make_particle((2.0, 3.0), (0.5, 0.1), (4.0, 5.0)), gbest)

    assert not np.allclose(first["velocity"], second["velocity"])


def test_r1_r2_per_dimension_not_shared(monkeypatch):
    """Per canonical PSO: r1, r2 are d-dimensional vectors, each component sampled independently"""
    pso = _make_pso()
    particle = _make_particle((2.0, 3.0), (0.5, 0.1), (4.0, 5.0))
    gbest = np.array((6.0, 7.0), dtype=float)
    recorded_sizes = []

    def fake_uniform(low=0.0, high=1.0, size=None):
        recorded_sizes.append(size)
        if len(recorded_sizes) == 1:
            return np.array((0.1, 0.9), dtype=float)
        return np.array((0.2, 0.8), dtype=float)

    monkeypatch.setattr(np.random, "uniform", fake_uniform)

    updated = pso.update_velocity(particle, gbest)

    assert recorded_sizes == [2, 2]
    expected = np.array((1.0, 1.0), dtype=float) * np.array((0.5, 0.1), dtype=float)
    expected += 0.1 * np.array((0.1, 0.9), dtype=float) * (np.array((4.0, 5.0), dtype=float) - np.array((2.0, 3.0), dtype=float))
    expected += 0.2 * np.array((0.2, 0.8), dtype=float) * (gbest - np.array((2.0, 3.0), dtype=float))

    assert np.allclose(updated["velocity"], expected)


def test_zero_velocity_when_at_attractor():
    """Per Shi & Eberhart 1998: when x == pbest == gbest, new velocity = ω*v_old only"""
    pso = _make_pso()
    particle = _make_particle((2.0, 3.0), (0.5, 0.3), (2.0, 3.0))
    gbest = np.array((2.0, 3.0), dtype=float)

    updated = pso.update_velocity(particle, gbest)

    np.testing.assert_allclose(updated["velocity"], np.array((0.5, 0.3), dtype=float))
