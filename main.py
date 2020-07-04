from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import sys
import math
import csv
import time


def takeSecond(element):
    return element[1]


def itembasedCF(testdata, pair_rateDict, user_businessDict, business_userDict):
    predictions = []
    for t in testdata:
        active_user = t[0]
        item = t[1]

        if active_user in user_businessDict and item in business_userDict:
            sum_a = 0
            for business in user_businessDict[active_user]:
                sum_a += pair_rateDict[(active_user, business)]
            avg_a = sum_a / len(user_businessDict[active_user])

            sum_i = 0
            for user in business_userDict[item]:
                sum_i += pair_rateDict[(user, item)]
            avg_i = sum_i / len(business_userDict[item])  # avg i: all rating

            weight_sum = 0
            numerator_sum = 0
            w_list = []
            for j in user_businessDict[active_user]:  # compare business i to business j
                sum_j = 0
                for u in business_userDict[j]:
                    sum_j += pair_rateDict[(u, j)]
                avg_j = sum_j / len(business_userDict[j])  # avg j: all rating

                co_rated = business_userDict[item] & business_userDict[j]  # users who co rated i and j
                if len(co_rated) > 0:
                    cosum_i = 0
                    cosum_j = 0
                    for co_user in co_rated:
                        cosum_i += pair_rateDict[(co_user, item)]
                        cosum_j += pair_rateDict[(co_user, j)]
                    coavg_i = cosum_i / len(co_rated)
                    coavg_j = cosum_j / len(co_rated)

                # pearson correlation of item i and j using all rating
                numerator = 0
                denominator_i = 0
                denominator_j = 0
                for co_user in co_rated:
                    numerator += (pair_rateDict[(co_user, item)] - avg_i) * (pair_rateDict[(co_user, j)] - avg_j)
                    denominator_i += (pair_rateDict[(co_user, item)] - avg_i) ** 2
                    denominator_j += (pair_rateDict[(co_user, j)] - avg_j) ** 2
                denominator = (denominator_i ** (1/2)) * (denominator_j ** (1/2))
                if denominator == 0:
                    w_ij = 0
                else:
                    w_ij = numerator / denominator
                w_list.append((j, w_ij))

            w_list.sort(key=takeSecond, reverse=True)


            if len(w_list) >= 50:
                for i in range(50):
                    numerator_sum += pair_rateDict[(active_user, w_list[i][0])] * abs(w_list[i][1])
                    weight_sum += abs(w_list[i][1])
            else:
                weight_sum == 0
                #for i in range(len(w_list)):
                #    numerator_sum += pair_rateDict[(active_user, w_list[i][0])] * abs(w_list[i][1])
                #    weight_sum += abs(w_list[i][1])

            if weight_sum == 0:
                predictions.append(((active_user, item), avg_a))
            else:
                if numerator_sum / weight_sum >= 5:
                    predictions.append(((active_user, item), 5.0))
                else:
                    predictions.append(((active_user, item), numerator_sum / weight_sum))

        elif active_user in user_businessDict and item not in business_userDict:
            sum_a = 0
            for business in user_businessDict[active_user]:
                sum_a += pair_rateDict[(active_user, business)]
            avg_a = sum_a / len(user_businessDict[active_user])
            predictions.append(((active_user, item), avg_a))

        elif active_user not in user_businessDict and item in business_userDict:
            sum_i = 0
            for user in business_userDict[item]:
                sum_i += pair_rateDict[(user, item)]
            avg_i = sum_i / len(business_userDict[item])
            predictions.append(((active_user, item), avg_i))

        else:
            predictions.append(((active_user, item), 3))
    return predictions


def userbasedCF(testdata, pair_rateDict, user_businessDict, business_userDict):
    predictions = []
    for t in testdata:
        active_user = t[0]
        item = t[1]

        if active_user in user_businessDict and item in business_userDict:
            sum_a = 0
            for business in user_businessDict[active_user]:
                sum_a += pair_rateDict[(active_user, business)]
            avg_a = sum_a / len(user_businessDict[active_user])

            weight_sum = 0
            numerator_sum = 0
            w_list = []
            for user in business_userDict[item]:
                sum_u = 0
                count_b = 0
                for business in user_businessDict[user]:
                    sum_u = sum_u + pair_rateDict[(user, business)]
                    count_b +=1
                #avg_uuu = sum_u / count_b
                sum_u_other = sum_u - pair_rateDict[(user, item)]
                avg_u = sum_u_other / (count_b - 1)

                set_u = user_businessDict[user]
                co_rated = set_u & user_businessDict[active_user]
                corated_num = len(co_rated)
                if len(co_rated) > 0:
                    sum_active = 0
                    sum_user = 0
                    for co_item in co_rated:
                        sum_active += pair_rateDict[(active_user, co_item)]
                        sum_user += pair_rateDict[(user, co_item)]
                    avg_active = sum_active / corated_num
                    avg_user = sum_user / corated_num

                # pearson correlation
                numerator = 0
                denominator_a = 0
                denominator_u = 0
                for co_item in co_rated:
                    numerator += (pair_rateDict[(active_user, co_item)] - avg_a) * (pair_rateDict[(user, co_item)] - avg_u)
                    denominator_a += (pair_rateDict[(active_user, co_item)] - avg_a) ** 2
                    denominator_u += (pair_rateDict[(user, co_item)] - avg_u) ** 2
                denominator = (denominator_a * denominator_u) ** (1/2)
                if denominator == 0:
                    w_au = 0
                else:
                    w_au = numerator / denominator

                #w_list.append((user, w_au))

                # prediction
                numerator_sum += (pair_rateDict[(user, item)] - avg_u) * w_au
                weight_sum += abs(w_au)

            if weight_sum == 0:
                predictions.append(((active_user, item), avg_a))
            else:
                if avg_a + numerator_sum / weight_sum >= 5:
                    predictions.append(((active_user, item), 5.0))
                else:
                    predictions.append(((active_user, item), avg_a + numerator_sum / weight_sum))

        elif active_user in user_businessDict and item not in business_userDict:
            sum_a = 0
            for business in user_businessDict[active_user]:
                sum_a += pair_rateDict[(active_user, business)]
            avg_a = sum_a / len(user_businessDict[active_user])
            predictions.append(((active_user, item), avg_a))

        elif active_user not in user_businessDict and item in business_userDict:
            sum_i = 0
            for u in business_userDict[item]:
                sum_i += pair_rateDict[(u, item)]
            avg_i = sum_i / len(business_userDict[item])
            predictions.append(((active_user, item), avg_i))

        else:
            predictions.append(((active_user, item), 3))
    return predictions


def main():
    start = time.time()
    sc = SparkContext('local[*]', sys.argv[0])
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    case_id = int(sys.argv[3])
    output_file = sys.argv[4]
    trainRDD = sc.textFile(train_file)
    testRDD = sc.textFile(test_file)

    # remove header
    train_header = trainRDD.first()
    trainRDD = trainRDD.filter(lambda x: x != train_header).map(lambda x: x.split(','))
    test_header = testRDD.first()
    testRDD = testRDD.filter(lambda x: x != test_header).map(lambda x: x.split(','))

    # train data to dictionary
    user_row = trainRDD.map(lambda x: (x[0], 1)).groupByKey().map(lambda x: x[0]).collect()
    n_user = len(user_row)
    user_dic = {user_row[i]: i for i in range(n_user)}  # 'user': i

    business_col = trainRDD.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).map(lambda x: x[0]).collect()
    n_business = len(business_col)
    business_dic = {business_col[j]: j for j in range(n_business)}

    user_business = trainRDD.map(lambda x: (user_dic[x[0]], business_dic[x[1]])).groupByKey().mapValues(set).collect()
    user_businessDict = {user[0]: user[1] for user in user_business}

    business_user = trainRDD.map(lambda x: (business_dic[x[1]], user_dic[x[0]])).groupByKey().mapValues(set).collect()
    business_userDict = {business[0]: business[1] for business in business_user}  # {business j: user i1, i2, i3, ...}

    # for model training
    ratingsRDD = trainRDD.map(lambda x: (user_dic[x[0]], business_dic[x[1]], float(x[2])))

    test_user = testRDD.map(lambda x: (x[0], 1)).groupByKey().map(lambda x: x[0]).collect()
    for user in test_user:
        if user not in user_dic:
            user_dic[user] = n_user
            n_user += 1
    index_user = {v: k for k, v in user_dic.items()}    # user i: 'user'

    test_business = testRDD.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).map(lambda x: x[0]).collect()
    for business in test_business:
        if business not in business_dic:
            business_dic[business] = n_business
            n_business += 1
    index_business = {v: k for k, v in business_dic.items()}    # business j: 'business'

    trueRatingRDD = testRDD.map(lambda x: ((user_dic[x[0]], business_dic[x[1]]), float(x[2]))).collect()
    true_rating = {x[0]: x[1] for x in trueRatingRDD}   # {(user i, business j): rating}

    if case_id == 1:
        # Model-based CF recommendation system
        rank = 10
        numIterations = 10
        model = ALS.train(ratingsRDD, rank, numIterations, 0.3)
        testdata1 = testRDD.map(lambda x: (user_dic[x[0]], business_dic[x[1]]))
        prediction = model.predictAll(testdata1).map(lambda r: ((r[0], r[1]), r[2])).collect()
        pred_dict = {(p[0][0], p[0][1]): p[1] for p in prediction}
        for key, value in true_rating.items():
            if key not in pred_dict:
                pred_dict[key] = 3.0
        predictions = [((key[0], key[1]), value) for key, value in pred_dict.items()]
        n = len(predictions)

    elif case_id == 2:
        # User-based CF recommendation system
        rating = ratingsRDD.collect()
        pair_rateDict = {(r[0], r[1]): r[2] for r in rating}
        testdata2 = testRDD.map(lambda x: (user_dic[x[0]], business_dic[x[1]]))
        predictions = testdata2.repartition(12).mapPartitions(lambda x: userbasedCF(x, pair_rateDict, user_businessDict, business_userDict)).collect()
        n = len(predictions)

    elif case_id == 3:
        # Item-based CF recommendation system
        rating = ratingsRDD.collect()
        pair_rateDict = {(r[0], r[1]): r[2] for r in rating}
        testdata3= testRDD.map(lambda x: (user_dic[x[0]], business_dic[x[1]]))
        predictions = testdata3.repartition(12).mapPartitions(lambda x: itembasedCF(x, pair_rateDict, user_businessDict, business_userDict)).collect()
        n = len(predictions)

    count_0_1 = 0
    count_1_2 = 0
    count_2_3 = 0
    count_3_4 = 0
    count_4_5 = 0
    sum = 0
    for predict in predictions:
        square = (predict[1] - true_rating[predict[0]]) ** 2
        sum += square
        if 0 <= abs(predict[1] - true_rating[predict[0]]) < 1:
            count_0_1 += 1
        elif 1 <= abs(predict[1] - true_rating[predict[0]]) < 2:
            count_1_2 += 1
        elif 2 <= abs(predict[1] - true_rating[predict[0]]) < 3:
            count_2_3 += 1
        elif 3 <= abs(predict[1] - true_rating[predict[0]]) < 4:
            count_3_4 += 1
        elif 4 <= abs(predict[1] - true_rating[predict[0]]):
            count_4_5 += 1
    print(">=0 and <1", count_0_1)
    print(">=1 and <2", count_1_2)
    print(">=2 and <3", count_2_3)
    print(">=3 and <4", count_3_4)
    print(">=4", count_4_5)
    RMSE = math.sqrt(sum / n)
    print("RMSE", RMSE)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['user_id', ' business_id', ' prediction'])
        for predict in predictions:
            row = [index_user[predict[0][0]]]
            row.append(index_business[predict[0][1]])
            row.append(predict[1])
            writer.writerow(row)

    end = time.time()
    print("duration", end - start)


if __name__ == "__main__":
    main()