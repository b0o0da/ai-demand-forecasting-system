from . import analysis, data, insights, stat_models, train


def main():
    data.prepare()
    train.run()
    stat_models.run()
    analysis.eda()
    analysis.results()
    insights.compute()


if __name__ == "__main__":
    main()
