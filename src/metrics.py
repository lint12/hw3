from src.dependency_parse import DependencyParse


def get_metrics(predicted: DependencyParse, labeled: DependencyParse) -> dict:
    # TODO: Your code here!
    print('===in get_metrics===')
    print('predicted: ', predicted)
    print('-----------------------------------------------')
    print('labeled: ', labeled)

    print('# of tokens in predicted: ',len(predicted.tokens))
    print('# of heads in predicted: ',len(predicted.heads))
    print('# of deprels in predicted: ',len(predicted.deprel))

    print('# of tokens in labeled: ',len(labeled.tokens))
    print('# of heads in labeled: ',len(labeled.heads))
    print('# of deprels in labeled: ',len(labeled.deprel))

    correct_heads = 0 
    correct_heads_and_deprels = 0 
    for idx in range(len(labeled.tokens)): 
        if predicted.heads[idx] == labeled.heads[idx]: 
            correct_heads += 1

        if (predicted.heads[idx] == labeled.heads[idx]) and (predicted.deprel[idx] == labeled.deprel[idx]): 
            correct_heads_and_deprels += 1


    return {
        "uas": correct_heads/len(labeled.tokens),
        "las": correct_heads_and_deprels/len(labeled.tokens),
    }
