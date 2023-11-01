total = 0
lastId = 0
sumOfItem = []
ItemsInBag = []
product_info = [['Book       |40', 40],
                ['Cocacola   |20', 20],
                ['Eraser     |10', 10],
                ['Pen        |15', 15],
                ['Scissors   |60', 60]]


def get_item():
    ItemsInBag.insert(0, lastId)
    sumOfItem.insert(0, product_info[lastId][1])
    print(ItemsInBag)
    print(sumOfItem)
    return product_info[lastId][0]


def clear_bag(item):
    try:
        ItemsInBag.pop(item)
        sumOfItem.pop(item)
    except IndexError:
        ItemsInBag.pop()
        sumOfItem.pop()
    print(ItemsInBag)
    print(sumOfItem)


def get_sum():
    total = 0
    for item in range(0, len(sumOfItem)):
        total = total + sumOfItem[item]
    return total


def add_to_list(label_id, label):
    global lastId
    if label_id != lastId:
        lastId = label_id
        item = product_info[label_id][0].split(" ", 1)
        label["text"] = item[0]
