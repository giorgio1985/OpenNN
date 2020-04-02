//   OpenNN: Open Neural Networks Library
 //   www.opennn.net
 //
 //   S T A T I S T I C S   H E A D E R
 //
 //   Artificial Intelligence Techniques, SL
 //   artelnics@artelnics.com
  
 #ifndef STATISTICS_H
 #define STATISTICS_H
  
 // System includes
  
 #include <math.h>
  
 // OpenNN includes
  
 #include "vector.h"
 #include "matrix.h"
 #include "tensor.h"
 #include "functions.h"
  
 using namespace std;
  
 namespace OpenNN
 {
  
  
  
 struct Descriptives {
  
   // Default constructor.
  
   explicit Descriptives();
  
   // Values constructor.
  
   explicit Descriptives(const double &, const double &, const double &, const double &);
  
  
   virtual ~Descriptives();
  
   // Set methods
  
   void set_minimum(const double &);
  
   void set_maximum(const double &);
  
   void set_mean(const double &);
  
   void set_standard_deviation(const double &);
  
   Vector<double> to_vector() const;
  
   bool has_minimum_minus_one_maximum_one();
  
   bool has_mean_zero_standard_deviation_one();
  
   void save(const string &file_name) const;
  
   void print(const string& = "Basic descriptives:") const;
  
  
   string name;
  
  
   double minimum = 0;
  
  
   double maximum = 0;
  
  
   double mean = 0;
  
  
   double standard_deviation = 0;
  
 };
  
  
  
  
 struct BoxPlot {
  
   // Default constructor.
  
   explicit BoxPlot() {}
  
   // Values constructor.
  
   explicit BoxPlot(const double &, const double &, const double &, const double &, const double &);
  
   virtual ~BoxPlot() {}
  
   double minimum;
  
   double first_quartile;
  
   double median;
  
   double third_quartile;
  
   double maximum;
 };
  
  
  
  
  
 struct Histogram
 {
  
   explicit Histogram();
  
  
   explicit Histogram(const size_t &);
  
  
   explicit Histogram(const Vector<double>&, const Vector<size_t>&);
  
  
   virtual ~Histogram();
  
   // Methods
  
   size_t get_bins_number() const;
  
   size_t count_empty_bins() const;
  
   size_t calculate_minimum_frequency() const;
  
   size_t calculate_maximum_frequency() const;
  
   size_t calculate_most_populated_bin() const;
  
   Vector<double> calculate_minimal_centers() const;
  
   Vector<double> calculate_maximal_centers() const;
  
   size_t calculate_bin(const double &) const;
  
   size_t calculate_frequency(const double &) const;
  
  
   Vector<double> centers;
  
  
   Vector<double> minimums;
  
  
   Vector<double> maximums;
  
  
   Vector<size_t> frequencies;
 };
  
  
      // Minimum
  
      double minimum(const Vector<double>&);
      size_t minimum(const Vector<size_t>&);
      time_t minimum(const Vector<time_t>&);
      double minimum(const Matrix<double>&);
      double minimum_missing_values(const Vector<double>&);
      Vector<double> columns_minimums(const Matrix<double>&, const Vector<size_t>& = Vector<size_t>());
      double minimum_matrix(const Matrix<double>& matrix);
  
      // Maximum
  
      double maximum(const Vector<double>&);
      size_t maximum(const Vector<size_t>&);
      time_t maximum(const Vector<time_t>&);
      double maximum(const Matrix<double>&);
      double maximum_missing_values(const Vector<double>&);
      Vector<double> columns_maximums(const Matrix<double>&, const Vector<size_t>& = Vector<size_t>());
      double maximum_matrix(const Matrix<double>& matrix);
  
      double strongest(const Vector<double>&);
  
  
      // Range
      double range(const Vector<double>&);
  
      // Mean
      double mean(const Vector<double>&);
      double mean(const Vector<double>&, const size_t&, const size_t&);
      double mean(const Matrix<double>&,  const size_t&);
      Vector<double> mean(const Tensor<double>&);
      Vector<double> mean(const Matrix<double>&, const Vector<size_t>&);
      Vector<double> mean(const Matrix<double>&, const Vector<size_t>&, const Vector<size_t>&);
      double mean_missing_values(const Vector<double>&);
      Vector<double> mean_missing_values(const Matrix<double>&, const Vector<size_t>&, const Vector<size_t>&);
      Vector<double> columns_mean(const Matrix<double>&);
      Vector<double> rows_means(const Matrix<double>&, const Vector<size_t>& );
  
      // Median
      double median(const Vector<double>&);
      double median(const Matrix<double>&, const size_t&);
      Vector<double> median(const Matrix<double>&);
      Vector<double> median(const Matrix<double>&, const Vector<size_t>&);
      Vector<double> median(const Matrix<double>&, const Vector<size_t>&, const Vector<size_t>&);
      double median_missing_values(const Vector<double>&);
      Vector<double> median_missing_values(const Matrix<double>&);
      Vector<double> median_missing_values(const Matrix<double>&, const Vector<size_t>&, const Vector<size_t>&);
  
      // Variance
      double variance(const Vector<double>&);
      double variance_missing_values(const Vector<double>&);
      Vector<double> explained_variance(const Vector<double>&);
  
      // Standard deviation
      double standard_deviation(const Vector<double>&);
      Vector<double> standard_deviation(const Vector<double>&, const size_t&);
      double standard_deviation_missing_values(const Vector<double>&);
  
      // Assymetry
      double asymmetry(const Vector<double>&);
      double asymmetry_missing_values(const Vector<double>&);
  
      // Kurtosis
      double kurtosis(const Vector<double>&);
      double kurtosis_missing_values(const Vector<double>&);
  
      // Quartiles
      Vector<double> quartiles(const Vector<double>&);
      Vector<double> quartiles_missing_values(const Vector<double>&);
  
      // Box plot
      BoxPlot box_plot(const Vector<double>&);
      BoxPlot box_plot_missing_values(const Vector<double>&);
      Vector<BoxPlot> box_plots(const Matrix<double>&, const Vector<Vector<size_t>>&, const Vector<size_t>&);
  
      // Descriptives vector
      Descriptives descriptives(const Vector<double>&);
      Descriptives descriptives_missing_values(const Vector<double>&);
  
      // Descriptives matrix
      Vector<Descriptives> descriptives(const Matrix<double>&);
      Vector<Descriptives> descriptives(const Tensor<double>&);
      Vector<Descriptives> descriptives(const Matrix<double>&, const Vector<size_t>&, const Vector<size_t>&);
      Vector<Descriptives> descriptives_missing_values(const Matrix<double>&);
      Vector<Descriptives> descriptives_missing_values(const Matrix<double>&, const Vector<size_t>&, const Vector<size_t>&);
  
      // Histograms
      Histogram histogram(const Vector<double>&, const size_t & = 10);
      Histogram histogram_missing_values(const Vector<double>&, const size_t & = 10);
      Histogram histogram_centered(const Vector<double>&, const double& = 0.0, const size_t & = 10);
      Histogram histogram(const Vector<bool>&);
      Histogram histogram(const Vector<int>&, const size_t & = 10);
      Vector<Histogram> histograms(const Matrix<double>&, const size_t& = 10);
      Vector<Histogram> histograms_missing_values(const Matrix<double>& matrix, const size_t& bins_number);
      Vector<size_t> total_frequencies(const Vector<Histogram>&);
  
      // Distribution
      size_t perform_distribution_distance_analysis(const Vector<double>&);
      size_t perform_distribution_distance_analysis_missing_values(const Vector<double>&, const Vector<size_t>&);
      double normal_distribution_distance(const Vector<double>&);
      double half_normal_distribution_distance(const Vector<double>&);
      double uniform_distribution_distance(const Vector<double>&);
  
      // Normality
      Vector<bool> perform_normality_analysis(const Vector<double>&);
      double normality_parameter(const Vector<double>&);
      bool perform_Lilliefors_normality_test(const Vector<double>&, const double&);
      Vector<bool> perform_Lilliefors_normality_test(const Vector<double>&, const Vector<double>&);
  
      // Minimal indices
      size_t minimal_index(const Vector<double>&);
      Vector<size_t> minimal_indices(const Vector<double>&, const size_t &);
      Vector<size_t> minimal_indices_omit(const Matrix<double>&, const double&);
      Vector<size_t> minimal_indices(const Matrix<double>&);
  
      // Maximal indices
      size_t maximal_index(const Vector<double>&);
      Vector<size_t> maximal_indices(const Vector<double>&, const size_t &);
      Vector<size_t> maximal_indices(const Matrix<double>&);
      Vector<size_t> maximal_indices_omit(const Matrix<double>&, const double&);
      Vector<double> variation_percentage(const Vector<double>&);
      double column_minimum(const size_t&);
      double column_maximum(const size_t&);
  
      // Means binary
      Vector<double> means_binary_column(const Matrix<double>&);
      Vector<double> means_binary_columns(const Matrix<double>&);
      Vector<double> means_binary_columns_missing_values(const Matrix<double>&);
  
      // Mean weights
      double weighted_mean(const Vector<double>&, const Vector<double>&);
      Vector<size_t> maximal_indices();
      Vector<Vector<size_t>> minimal_maximal_indices();
  
      // Percentiles
      Vector<double> percentiles(const Vector<double>&);
      Vector<double> percentiles_missing_values(const Vector<double>&);
  
      // Means by categories
      Vector<double> means_by_categories(const Matrix<double>& matrix);
      Vector<double> means_by_categories_missing_values(const Matrix<double>& matrix);
  
      // Means continuous
 }
  
 #endif // __STATISTICS_H