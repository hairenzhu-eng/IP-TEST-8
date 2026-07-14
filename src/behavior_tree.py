class BTStatus:
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class BTNode:
    def tick(self, blackboard):
        raise NotImplementedError


class BTCondition(BTNode):
    def __init__(self, predicate):
        self._predicate = predicate

    def tick(self, blackboard):
        return BTStatus.SUCCESS if self._predicate(blackboard) else BTStatus.FAILURE


class BTAction(BTNode):
    def __init__(self, action):
        self._action = action

    def tick(self, blackboard):
        return self._action(blackboard)


class BTSequence(BTNode):
    def __init__(self, *children):
        self._children = children

    def tick(self, blackboard):
        for child in self._children:
            status = child.tick(blackboard)
            if status != BTStatus.SUCCESS:
                return status
        return BTStatus.SUCCESS


class BTSelector(BTNode):
    def __init__(self, *children):
        self._children = children

    def tick(self, blackboard):
        for child in self._children:
            status = child.tick(blackboard)
            if status != BTStatus.FAILURE:
                return status
        return BTStatus.FAILURE
