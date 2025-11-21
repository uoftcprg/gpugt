def main():
    status = True
    arguments = {}

    while status:
        try:
            line = input()
        except EOFError:
            status = False
        else:
            index = line.index(' ')
            arguments[line[:index]] = line[index + 1:].strip()

    pot = arguments['-pot']
    board = arguments['-board']
    reach = arguments['-reach']

    print(
        (
            'universal_poker('
            'betting=nolimit,'
            'numPlayers=2,'
            'numRounds=4,'
            'blind=100 50,'
            'firstPlayer=2 1 1 1,'
            'numSuits=4,'
            'numRanks=13,'
            'numHoleCards=2,'
            'numBoardCards=0 3 1 1,'
            'stack=20000 20000,'
            'bettingAbstraction=fcpa,'
            f'potSize={pot},'
            f'boardCards={board},'
            f'handReaches={reach}'
            ')'
        ),
    )


if __name__ == '__main__':
    main()
