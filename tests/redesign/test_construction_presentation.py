import ast
from dataclasses import fields, is_dataclass
from functools import partial

import pytest

from indoor_vt.execution.explorer import method_entries
from indoor_vt.execution.presentation import construction, expression
from indoor_vt.execution.star_experiment import star_reconstructor


@pytest.mark.parametrize('ending', ['consumption', 'robust'])
def test_displayed_star_constructor_recreates_the_injected_objects(ending):
    p = dict(alpha=.7, half_window=2, neighbors=5, radius_m=.1, beta=2., tolerance_m=.1,
             support_fraction=.04, duplicate_m=.06)
    algorithm = star_reconstructor(p, ending)
    names = {'partial': partial}

    def collect(value):
        if isinstance(value, partial):
            collect(value.func)
            for v in list(value.args) + list((value.keywords or {}).values()):
                collect(v)
        elif is_dataclass(value):
            names[type(value).__name__] = type(value)
            for f in fields(value):
                collect(getattr(value, f.name))
        elif callable(value):
            names[value.__name__] = value
    collect(algorithm)
    displayed = expression(algorithm)
    restored = eval(displayed, {'__builtins__': {}, **names})
    assert expression(restored) == displayed
    assert 'grouper=StarGrouper()' in displayed
    receipt = {'method': 'mpurge_stars_'+ending, 'label': 'mpurge_stars_'+ending+'_ununified',
               'parameters': p, 'unify': False, 'frozen': {}, 'id': 'a'*64}
    entry, = method_entries(receipt)
    assert entry['selection'] == 'fixed' and entry['grouping'] == 'stars'
    assert not any(x.get('label') == 'ConnectedComponentsGrouper()' for x in entry['code']['Algorithm steps'])
    code = construction(receipt, {}, 'sources')['code']
    tree = ast.parse(code)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign)
    assert code == 'method = '+displayed
