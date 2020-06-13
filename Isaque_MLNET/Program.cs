using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Globalization;

namespace Isaque_MLNET
{
    class Program
    {
        static readonly string _irisPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.txt");
        static readonly string _iris2Path = Path.Combine(Environment.CurrentDirectory, "Data", "iris2.txt");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");

        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);
            IDataView dataView = mlContext.Data.LoadFromTextFile<IrisData>(_irisPath, hasHeader: false, separatorChar: ',');
            string featuresColumnName = "Features";
            var pipeline = mlContext.Transforms
                .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));

            var model = pipeline.Fit(dataView);
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }

            var predictor = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);

            NumberFormatInfo number = (NumberFormatInfo)CultureInfo.CurrentCulture.NumberFormat.Clone();
            number.NumberDecimalSeparator = ".";

            string[] iris2 = File.ReadAllLines(_iris2Path);
            foreach(string s in iris2)
            {
                string[] currentLine = s.Split(",");
                float[] values = new float[currentLine.Length - 1];
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = float.Parse(currentLine[i], number);
                }
                IrisData iris = new IrisData
                {
                    SepalLength = values[0],
                    SepalWidth = values[1],
                    PetalLength = values[2],
                    PetalWidth = values[3]
                };

                var prediction = predictor.Predict(iris);

                switch (prediction.PredictedClusterId)
                {
                    case 1:
                        Console.WriteLine("Predição foi Iris-setosa");
                        break;
                    case 2:
                        Console.WriteLine("Predição foi Iris-versicolor");
                        break;
                    case 3:
                        Console.WriteLine("Predição foi Iris-virginica");
                        break;
                }
            }

            Console.ReadKey();
        }
    }
}
