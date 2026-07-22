import re
from collections import defaultdict, deque
from typing import Set


class Go:

    REGEX_GO_ID = re.compile(r"(GO:\d{7})")

    def __init__(self, path: str):
        self.path = path

    def parse_go_obo(self):
        """Parse go.obo into a parent->children graph and an obsolete->replacement map."""
        parents_to_children = defaultdict(list)
        obsoletes = {}
        
        current_id = None
        parents = []
        replaced_ids = []

        def flush():
            if current_id is None:
                return
            if replaced_ids:
                obsoletes[current_id] = list(replaced_ids)
            for parent in parents:
                parents_to_children[parent].append(current_id)

        with open(self.path) as fd:
            for line in fd:
                line = line.strip()
                if line == "[Term]":
                    flush()
                    current_id = None
                    parents = []
                    replaced_ids = []
                elif line.startswith("id: GO:"):
                    current_id = line.split("id: ")[1]
                elif line.startswith("is_a:") and current_id:
                    parents.append(line.split("is_a: ")[1].split(" !")[0])
                elif line.startswith("replaced_by:"):
                    match = self.REGEX_GO_ID.search(line)
                    if match:
                        replaced_ids.append(match.group(1))
            flush()

        return parents_to_children, obsoletes

    @classmethod
    def get_all_children(cls, go_id, parents_to_children, obsoletes: Set = {}, add_go_id: bool = True):
        seen = set()
        to_visit = deque([go_id])
        while to_visit:
            current = to_visit.popleft()
            for child in parents_to_children.get(current, []):
                if child not in seen:
                    seen.add(child)
                    to_visit.append(child)
        if obsoletes:
            seen = Go.resolve_obsolete_go_ids(go_ids=seen, obsoletes=obsoletes)
        if add_go_id:
            seen.add(go_id)
        return seen

    @classmethod
    def resolve_obsolete_go_ids(cls, go_ids, obsoletes):
        """Add obsolete GO ids whose replacement (possibly a chain of replacements) is in go_ids."""
        resolved = set(go_ids)
        for obsolete_id, replaced_by in obsoletes.items():
            seen = set()
            queue = deque(replaced_by)
            while queue:
                candidate = queue.popleft()
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate in go_ids:
                    resolved.add(obsolete_id)
                    break
                queue.extend(obsoletes.get(candidate, []))
        return resolved
