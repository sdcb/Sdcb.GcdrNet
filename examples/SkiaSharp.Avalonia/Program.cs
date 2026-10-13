using Avalonia;

namespace SkiaSharp.Avalonia;

internal static class Program
{
    public static string StartupImagePath { get; private set; } = "";

    [STAThread]
    private static void Main(string[] args)
    {
        StartupImagePath = args.Length > 0
            ? Path.GetFullPath(args[0])
            : Path.Combine(AppContext.BaseDirectory, "example.jpg");
        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
    }

    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .LogToTrace();
}
