
import re
import sys

p_start_epoch = re.compile(r'Starting epoch (\d+)')
p_case = re.compile(r'/data/case_(\d{5})_x\.npy')

def extract_cases(log):
    cases_map = {}
    with open(log, 'r') as infile:
        epoch = 0
        for line in infile:
            if m := re.match(p_start_epoch, line):
                epoch = m.group(1)
                cases_map[epoch] = set()
            
            elif m := re.match(p_case, line):
                case = m.group(1)
                cases_map[epoch].add(case)

    return cases_map


def find_cases_in_commmon(log1, log2):

    log1_cases = extract_cases(log1)
    log2_cases = extract_cases(log2)


    for (epoch, set1), (_, set2) in zip(log1_cases.items(), log2_cases.items()):
        print(f'Epoch: {epoch}')
        print(f'Found {len(set1.intersection(set2))} cases in common:\n{set1.intersection(set2)}')
    


if __name__=='__main__':

    find_cases_in_commmon(sys.argv[1], sys.argv[2])