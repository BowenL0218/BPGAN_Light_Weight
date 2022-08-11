import heapq
import os
import math
import operator

def bindigits_huff(n, bits):
    s = bin(int(n) & int("1"*bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)

class HeapNode:
    def __init__(self, weight, freq):
        self.weight = weight
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        if(other == None):
            return -1
        if(not isinstance(other, HeapNode)):
            return -1
        return operator.lt(self.freq,other.freq)


class HuffmanCoding:
    def __init__(self, weights):
        self.weights = weights
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    # functions for compression:

    def make_frequency_dict(self, weights):
        frequency = {}
        for weight in weights:
            if not weight in frequency:
                frequency[weight] = 0
            frequency[weight] += 1
        return frequency

    def make_heap(self, frequency):
        for key in frequency:
            node = HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while(len(self.heap)>1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if(root == None):
            return

        if(root.weight != None):
            self.codes[root.weight] = current_code
            self.reverse_mapping[current_code] = root.weight
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        #print("root",root.weight)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def get_encoded_weights(self, weights):
        encoded_weights = [] 
        for weight in weights:
                encoded_weights.append(self.codes[weight])
        return encoded_weights

    def get_codes(self):
        return self.codes 

    def get_code_table(self):
        weight_table = []
        weight_table_subtree = ['','','','','','','','','','','','','','','','']
        weight_table.append(weight_table_subtree)
        for weight in self.codes:
            code_raw = self.codes[weight]
            code_packets = float(len(code_raw))/float(4)
            #print(weight,":",code_raw)
            #print(code_packets)
            sub_tree_index = 0
            #print('code: ' + str(code_raw))
            #store code raw into multiple tables
            for i in range(0, int(math.ceil(code_packets))):
                if (i * 4 + 4 >= len(code_raw)): #last packet
                    #print('weight: ' + str(code_raw[i*4:]))
                    #print('used bindigits: '+ str(bindigits_huff(len(code_raw[i*4:]), 5)))
                    #print('weight bindigits: '+ str(bindigits_huff(weight, 8)))
                    used_bits = '{message:>5{fill}}'.format(message=str(bindigits_huff(len(code_raw[i*4:]), 5)), fill='').replace(' ', '0')
                    weight_bits = '{message:>8{fill}}'.format(message=str(bindigits_huff(weight, 8)), fill='').replace(' ', '0')

                    for j in range (0, 2**(4-len(code_raw[i*4:]))):
                        weight_table[sub_tree_index][2**(4-len(code_raw[i*4:]))*int(code_raw[i*4:],2) + j] = '10' + used_bits + weight_bits 
                        #print('write: ' + str(2**(4-len(code_raw[i*4:])) + j) + ' ' + '1' + used_bits + weight_bits)
                else:
                    #first time
                    #print('weight: ' + str(code_raw[i*4: i*4+ 4]))
                    if(weight_table[sub_tree_index][int(code_raw[i*4: i*4+ 4], 2)] == ''):
                        weight_table.append(['','','','','','','','','','','','','','','','']) #add a new subtree
                        tree_ptr = '{message:>5{fill}}'.format(message=str(bindigits_huff(len(weight_table)-1, 5)), fill='').replace(' ', '0')
                        weight_table[sub_tree_index][int(code_raw[i*4: i*4+ 4], 2)] = '11' + tree_ptr + '0'*8 
                        #print('write ' + weight_table[sub_tree_index][int(code_raw[i*4: i*4+ 4], 2)])
                        sub_tree_index = len(weight_table)-1
                        #print('look up in subtree ' + str(sub_tree_index))
                    else:
                        #print('entree: ' + weight_table[sub_tree_index][int(code_raw[i*4: i*4+ 4], 2)])
                        sub_tree_index = int(weight_table[sub_tree_index][int(code_raw[i*4: i*4+ 4], 2)][2:2+5], 2)
        return weight_table 

    def compress(self):

        frequency = self.make_frequency_dict(self.weights)
        #print(frequency)

        self.make_heap(frequency)
        self.merge_nodes()
        self.make_codes()

        encoded_weights = self.get_encoded_weights(self.weights)

        #print("Compressed")
        return encoded_weights 

    """ functions for decompression: """

    def decode_weight(self, encoded_weight):
        decoded_weight = 0

        if(encoded_weight in self.reverse_mapping):
            decoded_weight = self.reverse_mapping[encoded_weight]

        return decoded_weight

    def decompress(self, encoded_weights):

        decompressed_weights = []
        for encoded_weight in encoded_weights:
            decompressed_weights.append(self.decode_weight(encoded_weight))
            
        #print("Decompressed")
        return decompressed_weights 
