from Project.analyzer import Analyzer

if __name__ == '__main__':
    analyzer = Analyzer()

    print("Calculating features")
    analyzer.calc_features()
    print("Save features")
    analyzer.save()
    print("Calculate and save features")
    analyzer.get_results()
