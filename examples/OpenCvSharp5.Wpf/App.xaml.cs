using System.IO;
using System.Windows;

namespace OpenCvSharp5.Wpf;

public partial class App : Application
{
    private void OnStartup(object sender, StartupEventArgs e)
    {
        string imagePath = e.Args.Length > 0
            ? Path.GetFullPath(e.Args[0])
            : Path.Combine(AppContext.BaseDirectory, "example.jpg");
        new MainWindow(imagePath).Show();
    }
}
