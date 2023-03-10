#!/usr/bin/env python3
# ============================================================================
#  femtospice.py -- Spice-like Circuit Simulator in Miniature
#
#  Copyright 2023 Scott Forbes
#
# Femtospice is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
# Femtospice is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
# You should have received a copy of the GNU General Public License along
# with Femtospice. If not, see <https://www.gnu.org/licenses/>.
# ============================================================================

import sys
import re
import time
import numpy as np
import sympy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import siunits as si

verbose = 0
prints = []
plots = []


def normalize_path(path):
    norm_path = path.copy()
    first = sorted(norm_path)[0]
    i = norm_path.index(first)
    if i != 0:
        norm_path = norm_path[i:] + norm_path[:i]
    name = ','.join(norm_path)
    return norm_path, name


def find_loops():
    """Find all unique loops in graph"""
    global loops, visited, visited_comps
    loops = {}
    for node in graph:
        visited = set()
        visited_comps = set()
        if verbose >= 2:
            print(f"find_loops: {node=}")
        search(node, None, [])


def search(curr, curr_comp=None, path=[], indent=''):
    """Depth-first search for loops, from node curr along path"""
    global loops, visited, visited_comps
    if verbose >= 2:
        print(f"{indent}search({curr=}, {curr_comp=}, {path=})")
    visited_save = visited.copy()
    visited_comps_save = visited_comps.copy()
    if verbose >= 2:
        print(f"{indent}{visited_save=}")
    visited.add(curr)
    if curr_comp:
        visited_comps.add(curr_comp)
    path.append(curr)
    if verbose >= 2:
        print(f"{indent}appended. {path=}")
    for neighbor, comp in graph[curr]:
        comp_edge_nm, _, _, _ = comps[comp]
        n_comps_edge = len(edges[comp_edge_nm])

        path1 = path.copy()
        path1.append(':' + comp)

        if verbose >= 2:
            print(
                f"{indent}{neighbor=}, {n_comps_edge=}  {path=} "
                f"{comp_edge_nm=} {visited=}, {visited_comps=}"
            )
        if neighbor not in visited:
            # keep searching deeper
            search(neighbor, comp, path1, indent + '  ')
        elif neighbor == path1[0] and comp not in visited_comps:
            # have completed loop
            if verbose >= 2:
                print(f"{indent}reached beginning {path1=}")
            norm_path, name = normalize_path(path1)
            rev_path = norm_path.copy()
            rev_path.reverse()
            _, rev_name = normalize_path(rev_path)
            if verbose >= 2:
                print(f"{indent}{name=} {rev_name=}")
            if not (loops.get(name) or loops.get(rev_name)):
                loops[name] = norm_path
                if verbose >= 2:
                    print(f"{indent}{loops=} {visited=}, {visited_comps=}")
            elif verbose >= 2:
                print(f"{indent}duplicate loop")
            visited = visited_save
            visited_comps = visited_comps_save
            return

    if verbose >= 2:
        print(f"{indent}done.")
    visited = visited_save
    visited_comps = visited_comps_save


def rm_overlapped_loops_pass():
    global loops

    for loop1_nm, loop1 in loops.items():
        n_loop1_nodes = len([x for x in loop1 if x[0] != ':'])
        if verbose >= 2:
            print(f"{loop1_nm}: {n_loop1_nodes} nodes")
        overlap = True
        for edge_nm, edge in loop_edges[loop1_nm].items():
            common_edge = False
            for loop2_nm, loop2 in loops.items():
                n_loop2_nodes = len([x for x in loop2 if x[0] != ':'])
                if verbose >= 2:
                    print(f"  {loop2_nm}: {n_loop2_nodes} nodes")
                if n_loop2_nodes > n_loop1_nodes:
                    # loop1 is smaller than loop2: can't overlap
                    if verbose >= 2:
                        print(f"    loop1 is smaller")
                    overlap = False
                    break
                if loop2_nm != loop1_nm:
                    if edge_nm in loop_edges[loop2_nm].keys():
                        # found a common edge
                        common_edge = True
                        if verbose >= 2:
                            print(f"    common edge")

            if not common_edge or not overlap:
                # some edge unique to loop1, can't overlap
                overlap = False
                if verbose >= 2:
                    print(f"  some edge unique")
                break
        if overlap:
            if verbose >= 2:
                print(f"  overlapped")
            loops.pop(loop1_nm)
            return 1
    return 0


def rm_overlapped_loops():
    global loop_edges, edge_loops
    if verbose >= 2:
        print("rm_overlapped_loops")
    loop_edges = {}
    edge_loops = {}
    for loop_name, loop in loops.items():
        # get every edge in loop, as triplet [node1, comp, node2]
        edges = {}
        loop_wrap = loop + [loop[0]]
        if verbose >= 2:
            print(f"{loop_wrap=}")
        for i in range(0, len(loop_wrap) - 2, 2):
            edge = loop_wrap[i : i + 3]
            if verbose >= 2:
                print(f"{edge=}")
            forward = 1
            if edge[0] > edge[2]:
                edge.reverse()
                forward = -1
            edge_name = ','.join(edge)
            edges[edge_name] = edge, forward
            if not edge_name in edge_loops:
                edge_loops[edge_name] = []
            if not loop_name in edge_loops[edge_name]:
                edge_loops[edge_name].append(loop_name)
        loop_edges[loop_name] = edges

    while rm_overlapped_loops_pass() > 0:
        pass
    if verbose >= 2:
        print(f"  {edge_loops=}")


def rk4_step(y0, h, f):
    """One step of 4th order Runge-Kutta integration"""
    k1 = h * f(y0)
    k2 = h * f(y0 + 0.5 * k1)
    k3 = h * f(y0 + 0.5 * k2)
    k4 = h * f(y0 + k3)
    return y0 + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class Figure:
    """A matplotlib figure window"""

    winx = None
    winy = None
    num = 1

    def __init__(self):
        # create a matplotlib plot
        self.figure = plt.figure(Figure.num, figsize=(6, 3.5))
        self.figure.clear()
        Figure.num += 1
        self.ax = plt.axes()
        self.max_x = 0.0
        self.ylabel = ""


class NFmtr(ticker.Formatter):
    def __init__(self, scale_x):
        self.scale_x = scale_x

    def __call__(self, x, pos=None):
        return "{0:g}".format(x / 10**self.scale_x)


def plot(plot_data):
    global title, plot_var_names
    print(f"Plotting {len(plot_data)} points")
    plt.rcParams.update({'mathtext.default': 'regular'})
    plt.rcParams.update({'font.size': 9})
    fig = Figure()
    plt.figure(fig.figure.number)
    fig.title = title

    times = []
    for row in plot_data:
        times.append(row[0])

    min_y = 1e15
    max_y = -1e15
    for i, name in enumerate(plot_var_names):
        ys = []
        for row in plot_data:
            value = row[i + 1]
            ys.append(value)
            min_y = min(value, min_y)
            max_y = max(value, max_y)
        plt.plot(times, ys.copy(), label=name)
    fig.min_y = min_y
    fig.max_y = max_y

    fig.xlabel = "time"
    fig.xunit = "s"
    fig.max_x = max(plot_data[-1][0], fig.max_x)

    fig.ylabel = f"voltage (%sV)"

    fm = plt.get_current_fig_manager()
    fm.set_window_title(title)

    plt.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
    x_si = si.si(fig.max_x)
    range_y = fig.max_y - fig.min_y
    y_si = si.si(range_y)
    units_x = x_si[-1]
    if units_x.isdigit():
        units_x = ''
    units_y = y_si[-1]
    if units_y.isdigit():
        units_y = ''
    scale_x = si.SIPowers.get(units_x)
    scale_y = si.SIPowers.get(units_y)
    if verbose >= 2:
        print(f"{fig.min_y=} {fig.max_y=} {range_y=} {y_si=}")
        print(f"{fig.ylabel=} {units_y=} {scale_y=}")
    if scale_x is not None:
        if verbose >= 2:
            print(f"Scaling plot X axis by {10**scale_x:g} ({units_x})")
        fig.ax.xaxis.set_major_formatter(NFmtr(scale_x))
    else:
        units_x = ''
    if scale_y is not None:
        fig.ax.yaxis.set_major_formatter(NFmtr(scale_y))
    else:
        units_y = ''
    plt.xlabel(f"{fig.xlabel} ({units_x}{fig.xunit})")
    plt.ylabel(fig.ylabel % units_y)

    plt.grid(True)
    plt.subplots_adjust(left=0.10, top=0.90, right=0.97, bottom=0.15)
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc='lower left',
        ncol=min(len(plots), 6),
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
    )
    plt.margins(0)
    ##plt.show(block=False)
    plt.show()


def eqns_source(x_comps, solution_eqns, offset):
    """generate python source for each component in x_comps"""
    source = ''
    loop_source = ''
    for i, x_comp in enumerate(x_comps):
        name = x_comp.name
        eqn = solution_eqns[i + offset]
        if name[0] == 'd':
            # a delta for a variable to be integrated: make RK4 function
            parm = name[1:]
            globs = [s.name for s in eqn.free_symbols if s.name != parm]
            source += (
                f"def {name}({parm}):\n"
                f"    global {', '.join(globs)}\n"
                f"    return {eqn}\n"
            )
        else:
            loop_source += f"    {name} = {eqn}\n"
    return source, loop_source


def signed_comp(comp, node):
    edge_nm, edge_dir, _, _ = comps[comp]
    edge_node1, _, edge_node2 = edge_nm.split(',')
    sign = 1 if edge_node1 == node else -1
    sign *= edge_dir
    if sign < 0:
        return f"(-v{comp})"
    return f"v{comp}"


def find_gnd(node, path):
    """breadth-first search for shortest path to node 0"""
    global visited

    if verbose >= 2:
        print(f"find_gnd({node}, {path})")
    con_nodes = graph[node]
    for node2, comp2 in con_nodes:
        if node2 == '0':
            path.append(signed_comp(comp2, node))
            if verbose >= 2:
                print(f"  found node 0 @ {path})")
            return path

    for node2, comp2 in con_nodes:
        if not node2 in visited:
            if verbose >= 2:
                print(f"  trying {node2}")
            path2 = path.copy()
            path2.append(signed_comp(comp2, node))
            path2 = find_gnd(node2, path2)
            if path2:
                return path2
        visited.add(node2)

    return None


def get_var_name(var, vi_comp_names):
    """Get a variable's printing name, adding a node eqn if needed"""
    global loop_source

    v_i = var[0]
    node = var[1:]
    name = f"{v_i}({node})"

    if v_i == 'v' and not var in vi_comp_names:
        if verbose >= 2:
            print(f"node '{var}' missing, building eqn")
        if not node in graph.keys():
            raise ValueError(f"unknown node '{node}'")
        vi_comp_names.append(var)

        # find a route to node 0 from this node for voltage
        visited = set()
        gnd_comps = find_gnd(node, [])
        if len(gnd_comps) == 0:
            raise ValueError(f"can't find ground path for node '{node}'")
        var_eqn = ' + '.join(gnd_comps)

        loop_source += f"    {var} = {var_eqn}\n"
    return name


def pwl(t, *args):
    """Piecewise linear interpolation of (time, voltage) pairs"""
    t0 = -1
    v0 = 0
    for i in range(0, len(args), 2):
        t1, v1 = args[i : i + 2]
        if t1 >= t:
            return v0 + (t - t0) * (v1 - v0) / (t1 - t0)
        t0 = t1
        v0 = v1
    return v1


def build_py_source():
    """generate python source for circuit simulation"""
    global prints, plots, plot_var_names, visited, loop_source
    n_comps = len(comps)
    if n_comps == 0:
        raise ValueError("No components in circuit")
    comps_list = list(comps.keys())
    n_loops = len(loops)
    if n_loops == 0:
        raise ValueError("No current loops in circuit")
    n_nodes = len(graph)
    if n_nodes == 0:
        raise ValueError("No nodes in circuit")

    # Kirchhoffs Voltage Law equations, one per loop
    KVL = np.zeros((n_loops, n_comps))
    # Kirchhoffs Current Law equations, one per node
    KCL = np.zeros((n_nodes, n_comps))
    if verbose >= 1:
        print(
            f"\nbuilding {n_comps} x {n_loops} KVL and "
            f"{n_comps} x {n_nodes} KCL matrix"
        )

    # traverse each current loop's edges, in order
    for i, loop_nm in enumerate(loops.keys()):
        edges1 = loop_edges[loop_nm]
        if verbose >= 1:
            print(f"i{i} loop '{loop_nm}' edges={edges1}")

        for edge, edge_dir in edges1.values():
            comp = edge[1][1:]
            _, comp_dir, _, _ = comps[comp]
            if verbose >= 2:
                print(f"  {comp}: {edge_dir} {comp_dir}")
            j = comps_list.index(comp)
            KVL[i, j] = edge_dir * comp_dir

    # get connections to each node
    nodes = list(graph.keys())
    nodes.sort()
    for i, node1 in enumerate(nodes):
        for (node2, comp) in graph[node1]:
            if verbose >= 2:
                print(f"{node1},{node2} {comp}:")
            edge2_name, comp_dir, _, _ = comps[comp]
            edge_dir = 1 if edge2_name == f"{node1},:{comp},{node2}" else -1
            if verbose >= 2:
                print(f"  {comp}: {edge2_name} {edge_dir} {comp_dir}")
            j = comps_list.index(comp)
            KCL[i, j] = edge_dir * comp_dir

    if verbose >= 1:
        print(f"\nKVL =\n  {'  '.join(comps_list)}\n{KVL}")
        print(f"\nKCL =\n{KCL}")

    # create component equations
    v, i = sympy.symbols("v,i")
    eqns = []
    v_comps = []
    i_comps = []
    dv_comps = []
    di_comps = []

    for comp, comp_data in comps.items():
        print(f"{comp}: {comp_data}")
        value = comp_data[2]
        v_comp = sympy.symbols(f"v{comp}")
        i_comp = sympy.symbols(f"i{comp}")
        v_comps.append(v_comp)
        i_comps.append(i_comp)
        ty = comp[0]
        if ty in 'c':
            dv_comp = sympy.symbols(f"dv{comp}")
            dv_comps.append(dv_comp)
        elif ty == 'l':
            di_comp = sympy.symbols(f"di{comp}")
            di_comps.append(di_comp)
        X = sympy.symbols(v_comp.name[1:].capitalize())
        if ty == 'v':
            eqn = v_comp - X
        elif ty == 'r':
            eqn = v_comp - X * i_comp
        elif ty == 'c':
            eqn = dv_comp - i_comp / X
        elif ty == 'l':
            eqn = di_comp - v_comp / X
        else:
            raise ValueError(f"Unknown component type for '{comp}'")
        if verbose >= 2:
            print(f" --> {eqn}")
        eqns.append(eqn)

    # create voltage equation strings from KVL
    for kvl_loop in KVL:
        eqn = 0
        for val, v_comp in zip(kvl_loop, v_comps):
            if val != 0:
                if val < 0:
                    eqn -= v_comp
                else:
                    eqn += v_comp
        eqns.append(eqn)

    # create current equation strings from KCL
    for kcl_sum in KCL:
        eqn = 0
        for val, i_comp in zip(kcl_sum, i_comps):
            if val != 0:
                if val < 0:
                    eqn -= i_comp
                else:
                    eqn += i_comp
        eqns.append(eqn)

    j = 0
    for i, vc in enumerate(v_comps):
        if vc.name[1] == 'c':
            v_comps[i] = dv_comps[j]
            j += 1

    j = 0
    for i, ic in enumerate(i_comps):
        if ic.name[1] == 'l':
            i_comps[i] = di_comps[j]
            j += 1

    vi_comps = v_comps + i_comps
    if verbose >= 1:
        print(f"\nsympy.linsolve({eqns}, {vi_comps})")
    solution = sympy.linsolve(eqns, vi_comps)
    if verbose >= 1:
        print(f"{solution=}")
    if len(solution) == 0:
        raise ValueError("No solution")
    solution_eqns = solution.args[0]

    # build circuit source code
    source = ''
    loop_source = ''
    for comp, (_, _, value, args) in comps.items():
        print(f"{comp} {value} {args}")
        ty = comp[0]
        sym = comp.capitalize()
        if ty in 'iv':
            if len(args) > 0:
                fn_args = [f"{si.floatSI(arg)}" for arg in args[1:]]
                loop_source += (
                    f"    {sym} = {args[0]}(t, {', '.join(fn_args)})\n"
                )
                continue
            else:
                for i, w in enumerate(value):
                    if w in ('ac', 'dc'):
                        value = si.floatSI(value[i + 1])
                        break
        if ty != 'd':
            source += f"{sym} = {abs(value):0.10g}\n"

    integ_comps = []
    for x_comp in vi_comps:
        name = x_comp.name
        if name[0] == 'd':
            var = name[1:]
            integ_comps.append(var)
            source += f"{var} = 0\n"

    ds, dls = eqns_source(v_comps, solution_eqns, 0)
    source += ds
    loop_source += dls
    ds, dls = eqns_source(i_comps, solution_eqns, len(v_comps))
    source += ds
    loop_source += dls

    for var in integ_comps:
        source += f"{var}_ = 0\n"

    vi_comp_names = [c.name for c in vi_comps]
    if prints == 'all':
        prints = vi_comp_names
    if plots == 'all':
        plots = vi_comp_names
    title2 = f"Transient Analysis  {time.asctime()}"

    if len(prints) > 0:
        var_names = var_fmts = ''
        for var in prints:
            v_i = var[0]
            node = var[1:]
            name = get_var_name(var, vi_comp_names)
            var_names += f" {name:13}"
            var_fmts += f"  {{{var}: 11.5e}}"

        source += f"""
print("\\n{' '*((80 - len(title))//2)}{title}")
print("{' '*((80 - len(title2))//2)}{title2}")
div_line = '-'*{21 + 14*len(prints)}
print(div_line)
print("Index     time         {var_names}")
print(div_line)
"""

    if len(plots) > 0:
        plot_var_names = []
        for var in plots:
            name = get_var_name(var, vi_comp_names)
            plot_var_names.append(name)
        source += "plot_data = []\n"

    source += f"""
nh = 100
h = dt / nh
i = 0
t = tstart
while t <= tstop:
{loop_source}"""

    if len(prints) > 0:
        source += f"""    if i % nh == 0:
        print(f"{{int(i/nh):<8d}} {{t: 11.5e}}{var_fmts}")
"""
    if len(plots) > 0:
        source += f"    plot_data.append([t, {', '.join(plots)}])\n"

    for var in integ_comps:
        source += f"    {var}_ = rk4_step({var}, h, d{var})\n"

    for var in integ_comps:
        source += f"    {var} = {var}_\n"

    source += """    t += h
    i += 1
"""

    if len(plots) > 0:
        source += "\nplot(plot_data)\n"

    if verbose >= 1:
        div_line = '-' * 56
        print(div_line)
        print(source, end='')
        print(div_line)
    return source


def do_tran(vars):
    print(f"tran {vars}")
    global dt, tstop, tstart, tmax
    dt, tstop, tstart, tmax = [si.floatSI(w) for w in vars]


def parse_sub(name, lines, subs=None, indent=''):
    """Parse a subcircuit"""
    global graph, comps, edges, subckts, prints, plots
    in_control = False

    if verbose >= 1 and name is not None:
        print(f"\n{indent}subckt {name}:")
    for orig_line in lines:
        line = orig_line.lower()
        words = re.split(r'[ \t]+', line)
        if verbose >= 2:
            print(f"{indent}{line}")
        cmd = words[0][:5]
        if cmd == '.cont':
            in_control = True
        elif cmd == '.endc':
            in_control = False
        elif cmd == '.end':
            break
        elif cmd == '.tran':
            do_tran(words[1:])
        elif in_control:
            if cmd == 'tran':
                do_tran(words[1:])
            elif cmd in ('print', 'plot'):
                args = []
                if words[1] == 'all':
                    args = 'all'
                else:
                    args = [f"{w[0]}{w[2:-1]}" for w in words[1:]]
                if verbose >= 2:
                    print(args)
                if words[0] == 'print':
                    prints = args
                else:
                    plots = args
            elif cmd == 'plot':
                do_plot(words[1:])
            elif cmd == 'echo':
                m = re.search(r'"([^"]*)"', orig_line)
                if m:
                    print(m[0][1:-1])
                else:
                    print()
        else:
            ty = cmd[0]
            if ty == 'x':
                call_args = words[1:-1]
                sub_name = words[-1]
                sub_args, sub_lines = subckts[sub_name]
                s = {}
                for k, v in zip(sub_args, call_args):
                    s[k] = v
                xname = cmd[1:]
                if name is not None:
                    xname = f"{name}{xname}"
                parse_sub(xname, sub_lines, s, indent + '  ')

            else:
                if len(words) < 4:
                    raise SyntaxError(f"component missing args: {line}")
                comp, node1, node2, value = words[:4]
                args = []
                if subs:
                    comp = f"{comp[0]}{name}_{comp[1:]}"
                    node1 = subs.get(node1, f"{name}_{node1}")
                    node2 = subs.get(node2, f"{name}_{node2}")

                ty = comp[0]
                if ty in 'iv':
                    # current or voltage source
                    parts = re.split(r'[()]', line)
                    if len(parts) > 1:
                        # optional args from inside parenthesis
                        words0 = re.split(r'[ \t]+', parts[0])
                        args = words0[-1:] + re.split(r'[ \t]+', parts[1])
                        # value: AC, DC keywords and values
                        value = words0[3:-1]
                        if len(parts) > 2 and len(p2 := parts[2].strip()) > 0:
                            value += re.split(r'[ \t]+', p2)
                    elif len(words) > 4:
                        # no parenthesis, but separate out AC, DC anyway
                        value = words[3:]
                        for i, w in enumerate(value):
                            if not (w[0].isdigit() or w in ('ac', 'dc')):
                                args = value[i:]
                                value = value[:i]
                                break

                elif ty in 'clr':
                    # passive
                    value = si.floatSI(value)

                if verbose >= 2:
                    print(f"{indent}-> {comp} {node1} {node2} {value}")
                if comp in comps:
                    raise ValueError(f"Duplicate component {comp}")

                edge_nm = f"{node1},:{comp},{node2}"
                edge_dir = 1
                if node1 > node2:
                    edge_dir = -1
                    if type(value) == type(1.0):
                        value = -value
                    edge_nm = f"{node2},:{comp},{node1}"
                # would like to use objects, but they're ~4 times slower
                comps[comp] = (edge_nm, edge_dir, value, args)
                if not edge_nm in edges:
                    edges[edge_nm] = []
                edges[edge_nm].append(comp)

                if node1 not in graph:
                    graph[node1] = []
                if node2 not in graph:
                    graph[node2] = []
                graph[node1].append((node2, comp))
                graph[node2].append((node1, comp))


def parse_deck(deck):
    """Parse a Spice file"""
    global graph, comps, edges, subckts, title
    graph = {}
    comps = {}
    edges = {}
    subckts = {}

    with open(deck) as f:
        lines = f.readlines()
        title = lines[0].strip()
        subckt_name = None
        args = None
        subckt_lines = []
        in_subckt = False
        lines2 = []

        # pass 1: gather subcircuits and model definitions
        for line in lines[1:]:
            line = line.split('*')[0].strip()
            if line == '':
                continue

            words = re.split("[\s\t]+", line.lower())
            cmd = words[0][:5]
            if cmd == '.subc':
                in_subckt = True
                subckt_name = words[1]
                args = words[2:]

            elif in_subckt:
                if cmd == '.ends':
                    in_subckt = False
                    subckts[subckt_name] = (args, subckt_lines)
                    subckt_lines = []
                else:
                    subckt_lines.append(line)

            else:
                lines2.append(line)

        if verbose >= 1:
            print("subckts:\n")
            for name, (args, lines) in subckts.items():
                print(f"{name}({args}) =")
                print("\n".join(lines))
                print()
            print()

        # pass 2: gather components and construct graph
        parse_sub(None, lines2)


def print_graph():
    print("\ngraph:")
    for name, nodes in graph.items():
        print(f"  {name}: ({', '.join(['|'.join(x) for x in nodes])})")


def print_comps():
    print("\ncomps:")
    for comp, (edge_name, edge_dir, value, args) in comps.items():
        if type(value) == type(1.0):
            value = si.si(value)
        print(f"  {comp} ({edge_name} {edge_dir} {value} {args})")


def compile_and_run(deck):
    """Compile a spice deck and run simulation"""
    parse_deck(deck)
    if verbose >= 1:
        print_graph()
        print_comps()

    find_loops()
    if verbose >= 2:
        print(' / '.join(loops.keys()))

    rm_overlapped_loops()
    if verbose >= 1:
        print("\ncurrent loops:")
        print(' / '.join(loops.keys()))

    source = build_py_source()
    exec(compile(source, 'source', 'exec'))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: femtospice <deck file> <verbose>")
        sys.exit(-1)

    deck = sys.argv[1]
    if len(sys.argv) > 2:
        verbose = int(sys.argv[2])

    compile_and_run(deck)
