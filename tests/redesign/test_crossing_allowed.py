import numpy as np
from indoor_vt.science.matching import MPUrgeMatcher, CrossingAllowedMPUrgeMatcher
from indoor_vt.execution.artifacts import identity
from indoor_vt.execution.explorer import method_entries


def test_only_crossing_veto_changes_with_same_endpoint_purging():
    rng = np.random.default_rng(97)
    added = 0
    for unify in (None, MPUrgeMatcher().size_unification_norm):
        for n, m in ((0, 7), (7, 0), (7, 7), (7, 11)):
            for _ in range(12):
                a, b = rng.uniform(0, 10, n), rng.uniform(0, 10, m)
                settings = dict(half_window=2, alpha=.7, size_unification_norm=unify)
                regular = MPUrgeMatcher(**settings).match(a, b)
                allowed = CrossingAllowedMPUrgeMatcher(**settings).match(a, b)
                np.testing.assert_array_equal(regular['trace'][:, :4], allowed['trace'][:, :4])
                np.testing.assert_array_equal(allowed['matches'], regular['trace'][:, :3])
                assert (allowed['trace'][:, 4] == 1).all()
                assert len(np.unique(allowed['matches'][:, 0])) == len(allowed['matches'])
                assert len(np.unique(allowed['matches'][:, 1])) == len(allowed['matches'])
                added += len(allowed['matches']) - len(regular['matches'])
    assert added > 0
    assert identity(MPUrgeMatcher()) != identity(CrossingAllowedMPUrgeMatcher())


def test_crossing_variant_has_distinct_names_tags_and_constructor_links():
    for method in ('mpurge_allow_crossing', 'mpurge_allow_crossing_consumption', 'mpurge_allow_crossing_robust'):
        for suffix in ('', '_lea', '_joint'):
            entries = method_entries(dict(method=method+suffix,label=method+suffix+'_unified',unify=True))
            assert entries
            for entry in entries:
                assert entry['allow_crossing'] and 'matching:crossing_allowed' in entry['tags']
                assert 'crossings allowed' in entry['source_full_name']
                if entry['component'] != 'initial':
                    links = [link for group in entry['code'].values() for link in group]
                    assert any(isinstance(link,dict) and link['label']=='CrossingAllowedMPUrgeMatcher()' for link in links)
