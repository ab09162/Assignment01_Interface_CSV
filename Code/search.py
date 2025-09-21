import csv
import heapq
import argparse
from math import gcd
from collections import defaultdict

# ------------------------ util (minimal subset) ------------------------
class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def push(self, item, priority):
        heapq.heappush(self.heap, (priority, self.counter, item))
        self.counter += 1

    def pop(self):
        prio, _, item = heapq.heappop(self.heap)
        return item, prio

    def isEmpty(self):
        return len(self.heap) == 0

# ------------------------ Search framework ------------------------
class SearchProblem:
    def getStartState(self):
        raise NotImplementedError

    def isGoalState(self, state):
        raise NotImplementedError

    def getSuccessors(self, state):
        raise NotImplementedError

    def getCostOfActions(self, actions):
        raise NotImplementedError

    def getHeuristic(self, state):
        return 0

# ------------------------ helper functions for cost management ------------------------

def is_tuple_cost(x):
    return isinstance(x, (tuple, list))

def make_zero_like(x):
    if is_tuple_cost(x):
        return tuple([0]*len(x))
    else:
        return 0

def add_costs(a, b):
    if is_tuple_cost(a) and is_tuple_cost(b):
        return tuple(x+y for x,y in zip(a,b))
    elif is_tuple_cost(a) and not is_tuple_cost(b):
        return tuple(a[i] + (b if i==0 else 0) for i in range(len(a)))
    elif not is_tuple_cost(a) and is_tuple_cost(b):
        return tuple(b[i] + (a if i==0 else 0) for i in range(len(b)))
    else:
        return a + b

def cost_lt(a, b):
    return a < b

# ------------------------ A* and Dijkstra implementations (record expansion order) ------------------------

def aStarSearch(problem, return_expanded=False):
    start = problem.getStartState()
    start_h = problem.getHeuristic(start)
    zero = make_zero_like(start_h)

    frontier = PriorityQueue()
    frontier.push((start, [], zero), add_costs(zero, start_h))

    best_g = {start: zero}
    expanded_order = []

    while not frontier.isEmpty():
        (state, path, g) , prio = frontier.pop()

        expanded_order.append(state)

        if problem.isGoalState(state):
            if return_expanded:
                return path, expanded_order
            return path

        if not cost_lt(g, best_g.get(state, (1e300 if not is_tuple_cost(g) else tuple([1e300]*len(g)))) ) and g != best_g.get(state):
            pass

        for succ, action, stepCost in problem.getSuccessors(state):
            new_g = add_costs(g, stepCost)
            old_best = best_g.get(succ)
            if old_best is None or cost_lt(new_g, old_best):
                best_g[succ] = new_g
                h = problem.getHeuristic(succ)
                priority = add_costs(new_g, h)
                frontier.push((succ, path + [action], new_g), priority)

    if return_expanded:
        return None, expanded_order
    return None


def dijkstraSearch(problem, return_expanded=False):
    start = problem.getStartState()
    h0 = problem.getHeuristic(start)
    zero = make_zero_like(h0)

    original_heuristic = problem.getHeuristic
    try:
        problem.getHeuristic = lambda s: zero
        res = aStarSearch(problem, return_expanded=return_expanded)
        return res
    finally:
        problem.getHeuristic = original_heuristic

# ------------------------ Problem I: Water Jug ------------------------
class WaterJugProblem(SearchProblem):
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z
        if Z > max(X, Y) or (Z % gcd(X, Y) != 0):
            raise ValueError(f"Target Z={Z} is not measurable with buckets {X},{Y}")

    def getStartState(self):
        return (0, 0)

    def isGoalState(self, state):
        a, b = state
        return a == self.Z or b == self.Z

    def getSuccessors(self, state):
        a, b = state
        succ = []
        if a != self.X:
            succ.append(((self.X, b), f"Fill A to {self.X}", 1))
        if b != self.Y:
            succ.append(((a, self.Y), f"Fill B to {self.Y}", 1))
        if a != 0:
            succ.append(((0, b), "Empty A", 1))
        if b != 0:
            succ.append(((a, 0), "Empty B", 1))
        if a != 0 and b != self.Y:
            transfer = min(a, self.Y - b)
            succ.append(((a - transfer, b + transfer), f"Pour A->B ({transfer})", 1))
        if b != 0 and a != self.X:
            transfer = min(b, self.X - a)
            succ.append(((a + transfer, b - transfer), f"Pour B->A ({transfer})", 1))

        return succ

    def getCostOfActions(self, actions):
        return len(actions)

    def getHeuristic(self, state):
        return 0

# ------------------------ Problem II: Route Planning ------------------------
class RoutePlanningProblem(SearchProblem):
    def __init__(self, connections_csv, heuristics_csv, tracktype_csv, startCity, goalCity):
        self.connections, self.cities = self._read_matrix_csv(connections_csv)
        self.tracktype, _ = self._read_matrix_csv(tracktype_csv)
        self.heuristics, _ = self._read_matrix_csv(heuristics_csv)

        if startCity not in self.cities or goalCity not in self.cities:
            raise ValueError("start or goal city not found in CSVs")
        self.start = startCity
        self.goal = goalCity

    def _read_matrix_csv(self, path):
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            header = [c.strip() for c in rows[0][1:]]
            data = {}
            for r in rows[1:]:
                rowcity = r[0].strip()
                data[rowcity] = {}
                for colname, cell in zip(header, r[1:]):
                    cell = cell.strip()
                    if cell == '' or cell == '-1':
                        data[rowcity][colname] = None
                    else:
                        try:
                            val = float(cell)
                            data[rowcity][colname] = val
                        except ValueError:
                            data[rowcity][colname] = cell
            return data, header

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state == self.goal

    def getSuccessors(self, state):
        succ = []
        for neigh, val in self.connections[state].items():
            if val is None:
                continue
            dist = float(val)
            ttype = self.tracktype.get(state, {}).get(neigh)
            jeep = 1 if (ttype == 'J') else 0
            stepCost = (dist, jeep)
            action = f"{state} -> {neigh} via {ttype or 'Unknown'} ({dist})"
            succ.append((neigh, action, stepCost))
        return succ

    def getCostOfActions(self, actions):
        total = (0.0, 0)
        for act in actions:
            try:
                dist_str = act[act.rfind('(')+1:act.rfind(')')]
                dist = float(dist_str)
            except Exception:
                dist = 0.0
            jeep = 1 if ('via J' in act or ' via J' in act or 'via J' in act) else 0
            total = add_costs(total, (dist, jeep))
        return total

    def getHeuristic(self, state):
        val = self.heuristics.get(state, {}).get(self.goal)
        if val is None:
            return (0.0, 0)
        else:
            return (float(val), 0)

# ------------------------ Main / Examples ------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['waterjug', 'route'], required=True)
    parser.add_argument('--x', type=int)
    parser.add_argument('--y', type=int)
    parser.add_argument('--z', type=int)
    parser.add_argument('--connections', type=str, default='/mnt/data/Connections.csv')
    parser.add_argument('--heuristics', type=str, default='/mnt/data/heuristics.csv')
    parser.add_argument('--tracktype', type=str, default='/mnt/data/TrackType.csv')
    parser.add_argument('--start', type=str)
    parser.add_argument('--goal', type=str)
    parser.add_argument('--show-expanded', action='store_true', help='Print the nodes expanded (in order) by A* and Dijkstra')
    args = parser.parse_args()

    if args.mode == 'waterjug':
        X = args.x
        Y = args.y
        Z = args.z
        if None in (X, Y, Z):
            raise SystemExit('For waterjug provide --x --y --z')
        print(f"Solving Water Jug: X={X}, Y={Y}, Z={Z}")
        prob = WaterJugProblem(X, Y, Z)
        if args.show_expanded:
            sol, expanded = aStarSearch(prob, return_expanded=True)
            print('Expanded nodes (A*):', expanded)
        else:
            sol = aStarSearch(prob)
        if sol is None:
            print('No solution found')
        else:
            print('Solution (actions):')
            for s in sol:
                print('  ', s)
            print('Cost (number of actions):', prob.getCostOfActions(sol))

    else:
        if None in (args.start, args.goal):
            raise SystemExit('For route provide --start and --goal')
        print(f"Solving Route from {args.start} to {args.goal}")
        prob = RoutePlanningProblem(args.connections, args.heuristics, args.tracktype, args.start, args.goal)
        print('-- Running A* (distance primary, minimize jeep segments secondary) --')
        if args.show_expanded:
            path, expanded = aStarSearch(prob, return_expanded=True)
            if path is None:
                print('No path found by A*')
            else:
                print('A* actions:')
                for a in path:
                    print(' ', a)
                print('A* cost (distance, jeep_count):', prob.getCostOfActions(path))
                print('A* expanded nodes in order:')
                print('  ' + ', '.join(map(str, expanded)))
        else:
            path = aStarSearch(prob)
            if path is None:
                print('No path found by A*')
            else:
                print('A* actions:')
                for a in path:
                    print(' ', a)
                print('A* cost (distance, jeep_count):', prob.getCostOfActions(path))

        print('-- Running Dijkstra (same cost model, zero heuristic) --')
        if args.show_expanded:
            path2, expanded2 = dijkstraSearch(prob, return_expanded=True)
            if path2 is None:
                print('No path found by Dijkstra')
            else:
                print('Dijkstra actions:')
                for a in path2:
                    print(' ', a)
                print('Dijkstra cost (distance, jeep_count):', prob.getCostOfActions(path2))
                print('Dijkstra expanded nodes in order:')
                print('  ' + ', '.join(map(str, expanded2)))
        else:
            path2 = dijkstraSearch(prob)
            if path2 is None:
                print('No path found by Dijkstra')
            else:
                print('Dijkstra actions:')
                for a in path2:
                    print(' ', a)
                print('Dijkstra cost (distance, jeep_count):', prob.getCostOfActions(path2))
                
                
"Ru"
