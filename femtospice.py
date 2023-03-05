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
import siunits as si

verbose = 0


def parse_deck(deck):
    """Parse Spice file into a graph, comps, and edges"""
    global graph, comps, edges, title, dt, tstop, tstart, tmax, printed_comps
    graph = {}
    comps = {}
    edges = {}

    with open(deck) as f:
        lines = f.readlines()
        title = lines[0].strip()
        in_control = False

        for line in lines[1:]:
            line = line.split('*')[0].strip()
            if verbose >= 3:
                print(f"'{line}'")
            words = re.split("[\s\t]+", line)
            if len(words) >= 1:
                cmd = words[0][:5]
                if cmd == '.cont':
                    in_control = True
                    continue
                elif cmd == '.endc':
                    in_control = False
                    continue
                elif cmd == '.tran':
                    dt, tstop, tstart, tmax = [
                        si.floatSI(w) for w in words[1:]
                    ]
                    continue
                elif cmd == 'print':
                    if words[1] == 'all':
                        printed_comps = 'all'
                    else:
                        printed_comps = [
                            f"{w[0]}{w[2:-1].lower()}" for w in words[1:]
                        ]
                    if verbose >= 2:
                        print(printed_comps)
                    continue
                elif words[0] == '' or line[0] == '.' or in_control:
                    continue
            if len(words) >= 4:
                comp, node1, node2, value_s = words[:4]
                comp = comp.lower()
                node1 = node1.lower()
                node2 = node2.lower()
                if comp[0] == 'v':
                    if value_s == 'pwl(0':
                        value_s = words[6][:-1]
                    elif value_s == 'ac':
                        value_s = words[4]
                value = si.floatSI(value_s)
                if comp[0] in "vrlc":
                    edge_nm = f"{node1},:{comp},{node2}"
                    edge_dir = 1
                    if node1 > node2:
                        edge_dir = -1
                        value = -value
                        edge_nm = f"{node2},:{comp},{node1}"
                    if comp in comps:
                        raise ValueError(f"Duplicate component {comp}")
                    comps[comp] = (edge_nm, edge_dir, value)
                    if not edge_nm in edges:
                        edges[edge_nm] = []
                    edges[edge_nm].append(comp)
                if node1 not in graph:
                    graph[node1] = []
                if node2 not in graph:
                    graph[node2] = []
                graph[node1].append((node2, comp))
                graph[node2].append((node1, comp))
            else:
                print(f"*** WARNING: unrecognized line '{line}'")

    if verbose >= 1:
        print(f"{edges=}")


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
        comp_edge, _, _ = comps[comp]
        n_comps_edge = len(edges[comp_edge])

        path1 = path.copy()
        path1.append(':' + comp)

        if verbose >= 2:
            print(
                f"{indent}{neighbor=}, {n_comps_edge=}  {path=} "
                f"{comp_edge=} {visited=}, {visited_comps=}"
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
    edge_nm, edge_dir, _ = comps[comp]
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


def build_py_source():
    """generate python source for circuit simulation"""
    global printed_comps, visited
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
            _, comp_dir, _ = comps[comp]
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
            edge2_name, comp_dir, _ = comps[comp]
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

    for comp in comps.keys():
        v_comp = sympy.symbols(f"v{comp}")
        i_comp = sympy.symbols(f"i{comp}")
        v_comps.append(v_comp)
        i_comps.append(i_comp)
        ty = comp[0]
        if ty == 'c':
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
        else:
            eqn = di_comp - v_comp / X
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
    solution = sympy.linsolve(eqns, vi_comps)
    if verbose >= 1:
        print(f"\nsympy.linsolve({eqns}, {vi_comps})")
        print(f"{solution=}")
    if len(solution) == 0:
        raise ValueError("No solution")
    solution_eqns = solution.args[0]

    # build circuit source code
    source = ''
    loop_source = ''
    for comp, (_, _, value) in comps.items():
        source += f"{comp.capitalize()} = {abs(value):0.10g}\n"

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
    if printed_comps == 'all':
        printed_comps = vi_comp_names

    var_names = var_fmts = ''
    for var in printed_comps:
        v_i = var[0]
        node = var[1:]
        name = f"{v_i}({node})"
        var_names += f" {name:13}"
        var_fmts += f"  {{{var}: 11.5e}}"

        if v_i == 'v' and not var in vi_comp_names:
            if verbose >= 2:
                print(f"node '{var}' missing, building eqn")
            if not node in graph.keys():
                raise ValueError(f"unknown node '{node}'")

            # find a route to node 0 from this node for voltage
            visited = set()
            gnd_comps = find_gnd(node, [])
            if len(gnd_comps) == 0:
                raise ValueError(f"can't find ground path for node '{node}'")
            var_eqn = ' + '.join(gnd_comps)

            loop_source += f"    {var} = {var_eqn}\n"

    title2 = f"Transient Analysis  {time.asctime()}"
    source += f"""
print("\\n{' '*((80 - len(title))//2)}{title}")
print("{' '*((80 - len(title2))//2)}{title2}")
div_line = '-'*{21 + 14*len(printed_comps)}
print(div_line)
print("Index     time         {var_names}")
print(div_line)

nh = 100
h = dt / nh
i = 0
t = tstart
tstop += dt
while t <= tstop:
{loop_source}
    if i % nh == 0:
        print(f"{{int(i/nh):<8d}} {{t: 11.5e}}{var_fmts}")
"""
    for var in integ_comps:
        source += f"    {var}_ = rk4_step({var}, h, d{var})\n"

    for var in integ_comps:
        source += f"    {var} = {var}_\n"

    source += """    t += h
    i += 1
"""

    if verbose >= 1:
        div_line = '-' * 56
        print(div_line)
        print(source, end='')
        print(div_line)
    return source


def print_graph():
    print("\ngraph:")
    for name, nodes in graph.items():
        print(f"  {name}: ({', '.join([':'.join(x) for x in nodes])})")


def print_comps():
    print("\ncomps:")
    for comp, (edge_name, edge_dir, value) in comps.items():
        print(f"  {comp} {edge_name} {edge_dir} {si.si(value)}")


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
