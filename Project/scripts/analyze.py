from Project.analyzer import Analyzer

if __name__ == '__main__':
    analyzer = Analyzer()

    analyzer.calc_features()
    analyzer.save()
    analyzer.get_results()
