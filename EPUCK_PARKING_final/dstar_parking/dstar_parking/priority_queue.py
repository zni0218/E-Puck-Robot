class Priority:
    """
    Representa uma prioridade com duas chaves (k1, k2).

    Isto é usado no D* Lite para ordenar células de forma lexicográfica:
    - primeiro compara k1
    - se k1 for igual, compara k2

    Quanto menor a prioridade, mais cedo o nó é processado.
    """

    def __init__(self, k1, k2):
        # k1: chave principal (ex: custo estimado + heurística)
        # k2: chave secundária (desempate / estabilidade)
        self.k1 = k1
        self.k2 = k2

    def __lt__(self, other):
        """
        Define ordem "menor que" entre prioridades.

        Usado pelo heap para decidir qual elemento sai primeiro.
        """
        return self.k1 < other.k1 or (self.k1 == other.k1 and self.k2 < other.k2)

    def __le__(self, other):
        """
        Define ordem "menor ou igual".

        Necessário para operações internas do heap.
        """
        return self.k1 < other.k1 or (self.k1 == other.k1 and self.k2 <= other.k2)


class PriorityNode:
    """
    Wrapper que junta:
    - uma célula do mapa (vertex)
    - uma prioridade associada

    Isto permite armazenar nós diretamente no heap.
    """

    def __init__(self, priority, vertex):
        # priority: objeto Priority (k1, k2)
        # vertex: célula ou nó do grafo (ex: (x, y))
        self.priority = priority
        self.vertex = vertex

    def __le__(self, other):
        # compara nós com base na prioridade
        return self.priority <= other.priority

    def __lt__(self, other):
        # compara nós com base na prioridade
        return self.priority < other.priority


class PriorityQueue:
    """
    Implementação de uma fila de prioridade (min-heap).

    Usada pelo D* Lite para:
    - escolher qual célula processar a seguir
    - manter ordenação eficiente por prioridade lexicográfica
    """

    def __init__(self):
        # heap principal (estrutura em árvore binária)
        self.heap = []

        # lista auxiliar para saber rapidamente se um vértice está na heap
        self.vertices_in_heap = []

    def top(self):
        # devolve o vértice com maior prioridade (menor custo)
        return self.heap[0].vertex

    def top_key(self):
        # devolve a prioridade do topo da heap
        # se vazio, retorna infinito (evita erros no algoritmo)
        if len(self.heap) == 0:
            return Priority(float('inf'), float('inf'))
        return self.heap[0].priority

    def pop(self):
        """
        Remove e devolve o elemento com menor prioridade.

        Implementação baseada no heap padrão (heapq),
        mas adaptada para guardar também vertices.
        """
        lastelt = self.heap.pop()  # último elemento da heap
        self.vertices_in_heap.remove(lastelt.vertex)

        if self.heap:
            returnitem = self.heap[0]
            self.heap[0] = lastelt
            self._siftup(0)
        else:
            returnitem = lastelt

        return returnitem

    def insert(self, vertex, priority):
        """
        Insere um novo vértice na heap com uma dada prioridade.
        """
        item = PriorityNode(priority, vertex)

        # regista que o vértice está na heap
        self.vertices_in_heap.append(vertex)

        # adiciona no heap e corrige posição
        self.heap.append(item)
        self._siftdown(0, len(self.heap) - 1)

    def remove(self, vertex):
        """
        Remove um vértice específico da heap.

        Usado quando o D* Lite decide que uma célula já não é necessária.
        """
        self.vertices_in_heap.remove(vertex)

        for index, priority_node in enumerate(self.heap):
            if priority_node.vertex == vertex:

                # substitui pelo último elemento
                self.heap[index] = self.heap[-1]
                self.heap.pop()
                break

        # reconstroi heap para manter consistência
        self.build_heap()

    def update(self, vertex, priority):
        """
        Atualiza a prioridade de um vértice já existente na heap.
        """
        for index, priority_node in enumerate(self.heap):
            if priority_node.vertex == vertex:
                self.heap[index].priority = priority
                break

        # reorganiza heap após mudança de prioridade
        self.build_heap()

    def build_heap(self):
        """
        Reconstrói a heap inteira de forma eficiente (bottom-up).
        """
        n = len(self.heap)

        for i in reversed(range(n // 2)):
            self._siftup(i)

    def _siftdown(self, startpos, pos):
        """
        Ajusta um elemento para cima na heap (subida).

        Usado quando se insere um novo elemento.
        """
        newitem = self.heap[pos]

        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self.heap[parentpos]

            if newitem < parent:
                self.heap[pos] = parent
                pos = parentpos
                continue
            break

        self.heap[pos] = newitem

    def _siftup(self, pos):
        """
        Ajusta um elemento para baixo na heap (descida).

        Usado quando se remove ou reestrutura a heap.
        """
        endpos = len(self.heap)
        startpos = pos
        newitem = self.heap[pos]

        childpos = 2 * pos + 1

        while childpos < endpos:

            rightpos = childpos + 1

            if rightpos < endpos and not self.heap[childpos] < self.heap[rightpos]:
                childpos = rightpos

            self.heap[pos] = self.heap[childpos]
            pos = childpos
            childpos = 2 * pos + 1

        self.heap[pos] = newitem

        self._siftdown(startpos, pos)