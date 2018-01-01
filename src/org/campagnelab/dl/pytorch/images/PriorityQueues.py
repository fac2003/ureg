from queue import PriorityQueue

def qfull(q):
    return q.maxsize > 0 and q.qsize() == q.maxsize

class PriorityQueues:
    def __init__(self, training_losses,max_queue_size):
        self.priority_queues = {}
        for tl in training_losses:
            # one priority queue per training loss:
            self.priority_queues[tl] = PriorityQueue(maxsize=max_queue_size)

    def put(self, training_loss, probability, unsup_example_index):
        queue=self.priority_queues[training_loss]
        # we dequeue the lowest probability when the queue is full, thus keeping the largest probability
        # values in the queue.
        if qfull(queue):
            queue.get()
        queue.put((probability, unsup_example_index))

    def get(self, training_loss):
        list=[]
        queue=self.priority_queues[training_loss]
        while not queue.empty() :
           probability, unsup_example_index=queue.get()
           print("Retrieved: prob={} index={}".format(probability, unsup_example_index))
           list.append(unsup_example_index)

        return list