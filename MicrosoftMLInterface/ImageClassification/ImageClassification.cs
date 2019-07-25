using ImageClassification.ModelScorer;
using System;
using System.IO;
using Microsoft.ML;
using ImageClassification.ImageDataStructures;


namespace ImageClassification
{
    public class Program
    {
        static void Main(string[] args)
        {
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            var tagsTsv = Path.Combine(assetsPath, "inputs", "zakladki", "image_list.tsv");
            var imagesFolder = Path.Combine(assetsPath, "inputs", "zakladki", "images");  
            var labelsTxt = Path.Combine(assetsPath, "inputs", "zakladki", "labels.txt");
            var pathToModel = @"D:\Files\GitHub\BinaryImageClassifier\BinaryImageClassifier\models\mobileNetV2\1564031768";

            try
            {
                var modelScorer = new TFModelScorer(tagsTsv, imagesFolder, pathToModel, labelsTxt);
                modelScorer.Score();

            }
            catch (Exception ex)
            {
                ConsoleHelpers.ConsoleWriteException(ex.ToString());
            }

            ConsoleHelpers.ConsolePressAnyKey();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            string fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }
    }
}
