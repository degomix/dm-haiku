# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts Haiku functions to dot."""

import collections
from collections.abc import Callable
import contextlib
import functools
import html
from typing import Any, NamedTuple

from haiku._src import data_structures
from haiku._src import module
from haiku._src import utils
import jax
import jax.core as jax_core
from jax._src.linear_util import wrap_init, transformation

try:
    import tree  # pylint: disable=g-import-not-at-top
except ImportError:
    tree = None

graph_stack = data_structures.ThreadLocalStack['Graph']()
Node = collections.namedtuple('Node', 'id,title,outputs')
Edge = collections.namedtuple('Edge', 'a,b')


class Graph(NamedTuple):
    title: str
    nodes: list[Node]
    edges: list[Edge]
    subgraphs: list['Graph']

    @classmethod
    def create(cls, title: str | None = None):
        return Graph(title=title, nodes=[], edges=[], subgraphs=[])

    def evolve(self, **kwargs) -> 'Graph':
        return Graph(**{**self._asdict(), **kwargs})


def to_dot(fun: Callable[..., Any]) -> Callable[..., str]:
    graph_fun = to_graph(fun)
    @functools.wraps(fun)
    def wrapped_fun(*args) -> str:
        return _graph_to_dot(*graph_fun(*args))
    return wrapped_fun


def abstract_to_dot(fun: Callable[..., Any]) -> Callable[..., str]:
    @functools.wraps(fun)
    def wrapped_fun(*args) -> str:
        dot_out = ''
        def dot_extractor_fn(*inner_args):
            nonlocal dot_out
            dot_out = to_dot(fun)(*inner_args)
        jax.eval_shape(dot_extractor_fn, *args)
        assert dot_out, 'Failed to extract dot graph from abstract evaluation'
        return dot_out
    return wrapped_fun


def name_or_str(o):
    return getattr(o, '__name__', str(o))


def to_graph(fun):
    @functools.wraps(fun)
    def wrapped_fun(*args):
        f = wrap_init(fun)
        args_flat, in_tree = jax.tree.flatten((args, {}))
        flat_fun, out_tree = jax.api_util.flatten_fun(f, in_tree)
        graph = Graph.create(title=name_or_str(fun))

        @contextlib.contextmanager
        def method_hook(mod: module.Module, method_name: str):
            subg = Graph.create()
            with graph_stack(subg):
                yield
            title = mod.module_name
            if method_name != '__call__':
                title += f' ({method_name})'
            graph_stack.peek().subgraphs.append(subg.evolve(title=title))

        with graph_stack(graph), module.hook_methods(method_hook):
            tag = jax.core.TraceTag()
            out_flat = _interpret_subtrace(tag).call_wrapped(*args_flat)
        out = jax.tree.unflatten(out_tree(), out_flat)

        return graph, args, out

    return wrapped_fun


@transformation
def _interpret_subtrace(tag, *in_vals):
    with jax.core.take_current_trace() as parent_trace:
        trace = DotTrace(parent_trace, tag)
        with jax.core.set_current_trace(trace):
            in_tracers = [DotTracer(trace, val) for val in in_vals]
            outs = yield in_tracers, {}
            yield [trace.to_val(t) for t in outs]


class DotTracer(jax.core.Tracer):
    def __init__(self, trace, val):
        self._trace = trace
        self.val = val

    @property
    def aval(self):
        return jax.core.get_aval(self.val)

    def full_lower(self):
        return self


class DotTrace(jax.core.Trace):
    def __init__(self, parent_trace, tag):
        super().__init__()
        self.parent_trace = parent_trace
        self.tag = tag

    def to_val(self, val):
        if isinstance(val, DotTracer) and val._trace.tag is self.tag:
            return val.val
        else:
            return val

    def process_primitive(self, primitive, tracers, params):
        vals = [self.to_val(t) for t in tracers]
        val_out = primitive.bind_with_trace(self.parent_trace, vals, params)
        if primitive is jax_core.primitives.jit_p:
            f = jax_core.jaxpr_as_fun(params['jaxpr'])
            f.__name__ = params['name']
            fun = wrap_init(f)
            return self.process_call(primitive, fun, tracers, params)

        outputs = list(jax.tree.leaves(val_out))
        if outputs:
            graph = graph_stack.peek()
            node = Node(id=outputs[0], title=str(primitive), outputs=outputs)
            graph.nodes.append(node)
            graph.edges.extend([(i, outputs[0]) for i in vals])

        return jax.tree_util.tree_map(lambda v: DotTracer(self, v), val_out)

    def process_call(self, call_primitive, f, tracers, params):
        assert call_primitive.multiple_results
        if (call_primitive in (jax_core.primitives.jit_p,) and params.get('inline', False)):
            f = _interpret_subtrace(f, self.tag)
            with jax.core.set_current_trace(self.parent_trace):
                vals_out = f.call_wrapped(*[self.to_val(t) for t in tracers])
                return [DotTracer(self, v) for v in vals_out]

        graph = Graph.create(title=f'{call_primitive} ({name_or_str(f.f)})')
        graph_stack.peek().subgraphs.append(graph)
        with graph_stack(graph):
            f = _interpret_subtrace(f, self.tag)
            with jax.core.set_current_trace(self.parent_trace):
                vals_out = f.call_wrapped(*[self.to_val(t) for t in tracers])
                return [DotTracer(self, v) for v in vals_out]

    process_map = process_call

    def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *, symbolic_zeros):
        del primitive, jvp, symbolic_zeros
        with jax.core.set_current_trace(self.parent_trace):
            return fun.call_wrapped(*tracers)

    def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees, symbolic_zeros):
        del primitive, fwd, bwd, out_trees, symbolic_zeros
        with jax.core.set_current_trace(self.parent_trace):
            return fun.call_wrapped(*tracers)


def _format_val(val):
    if not hasattr(val, 'shape'):
        return repr(val)
    shape = ','.join(map(str, val.shape))
    dtype = utils.simple_dtype(val.dtype)
    return f'{dtype}[{shape}]'


def escape(value):
    return html.escape(str(value))


def _max_depth(g: Graph) -> int:
    if g.subgraphs:
        return 1 + max(0, *[_max_depth(s) for s in g.subgraphs])
    else:
        return 0


def _scaled_font_size(depth: int) -> int:
    return int(1.4**depth * 14)


def _graph_to_dot(graph: Graph, args, outputs) -> str:
    if tree is None:
        raise ImportError('hk.to_dot requires dm-tree>=0.1.1.')

    def format_path(path):
        if isinstance(outputs, tuple):
            out = f'output[{path[0]}]'
            if len(path) > 1:
                out += ': ' + '/'.join(map(str, path[1:]))
        else:
            out = 'output'
            if path:
                out += ': ' + '/'.join(map(str, path))
        return out

    lines = []
    used_argids = set()
    argid_usecount = collections.Counter()
    op_outids = set()
    captures = []
    argids = {id(v) for v in jax.tree.leaves(args)}
    outids = {id(v) for v in jax.tree.leaves(outputs)}
    outname = {id(v): format_path(p) for p, v in tree.flatten_with_path(outputs)}

    def render_graph(g: Graph, parent: Graph | None = None, depth: int = 0):
        if parent:
            lines.extend([
                f'subgraph cluster_{id(g)} {{',
                '  style="rounded,filled";',
                '  fillcolor="#F0F5F5";',
                '  color="#14234B;";',
                '  pad=0.1;',
                f'  fontsize={_scaled_font_size(depth)};',
                f'  label = <<b>{escape(g.title)}</b>>;',
                '  labelloc = t;',
            ])

        for node in g.nodes:
            label = f'<b>{escape(node.title)}</b>'
            for o in node.outputs:
                label += '<br/>' + _format_val(o)
                op_outids.add(id(o))

            node_id = id(node.id)
            if node_id in outids:
                label = f'<b>{escape(outname[node_id])}</b><br/>' + label
                color = '#0053D6'
                fillcolor = '#AABFFF'
                style = 'filled,bold'
            else:
                color = '#FFDB13'
                fillcolor = '#FFF26E'
                style = 'filled'

            lines.append(
                f'{node_id} [label=<{label}>, id="node{node_id}", shape=rect, style="{style}", tooltip=" ", fontcolor="black", color="{color}", fillcolor="{fillcolor}"];'
            )

        for s in g.subgraphs:
            render_graph(s, parent=g, depth=depth - 1)

        if parent:
            lines.append(f'}}  // subgraph cluster_{id(g)}')

        for a, b in g.edges:
            if id(a) not in argids and id(a) not in op_outids:
                captures.append(a)
            a_id, b_id = map(id, (a, b))
            if a_id in argids:
                i = argid_usecount[a_id]
                argid_usecount[a_id] += 1
                lines.append(f'{a_id}{i} -> {b_id};')
            else:
                lines.append(f'{a_id} -> {b_id};')
            used_argids.add(a_id)

    graph_depth = _max_depth(graph)
    render_graph(graph, parent=None, depth=graph_depth)

    for path, value in tree.flatten_with_path(args):
        if value is None:
            continue
        node_id = id(value)
        if node_id not in used_argids:
            continue
        for i in range(argid_usecount[node_id]):
            label = f'<b>args[{escape(path[0])}]'
            if len(path) > 1:
                label += ': ' + '/'.join(map(str, path[1:]))
            label += '</b>'
            if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                label += f'<br/>{escape(_format_val(value))}'
            fillcolor = '#FFDEAF'
            fontcolor = 'black'
            if i > 0:
                label = '<b>(reuse)</b><br/>' + label
                fillcolor = '#FFEACC'
                fontcolor = '#565858'
            lines.append(
                f'{node_id}{i} [label=<{label}>, id="node{node_id}{i}", shape=rect, style="filled", fontcolor="{fontcolor}", color="#FF8A4F", fillcolor="{fillcolor}"];'
            )

    for value in captures:
        node_id = id(value)
        if (not hasattr(value, 'aval') and hasattr(value, 'size') and value.size == 1):
            label = f'<b>{value.item()}</b>'
        else:
            label = f'<b>{escape(_format_val(value))}</b>'
        lines.append(
            f'{node_id} [label=<{label}>, shape=rect, style="filled", fontcolor="black", color="#A261FF", fillcolor="#E6D6FF"];'
        )

    head = [
        'digraph G {',
        'rankdir = TD;',
        'compound = true;',
        f'label = <<b>{escape(graph.title)}</b>>;',
        f'fontsize={_scaled_font_size(graph_depth)};',
        'labelloc = t;',
        'stylesheet = <',
        '  data:text/css,',
        '  @import url(https://fonts.googleapis.com/css?family=Roboto:400,700);',
        '  svg text {',
        '    font-family: \'Roboto\';',
        '  }',
        '  .node text {',
        '    font-size: 12px;',
        '  }',
        '>'
    ]
    for node_id, use_count in argid_usecount.items():
        if use_count == 1:
            continue
        for a in range(use_count):
            for b in range(use_count):
                if a == b:
                    head.append(f'%23node{node_id}{a}:hover {{ stroke-width: 0.2em; }}')
                else:
                    head.append(f'%23node{node_id}{a}:hover ~ %23node{node_id}{b} {{ stroke-width: 0.2em; }}')
    lines.append('} // digraph G')
    return '\n'.join(head + lines) + '\n'
