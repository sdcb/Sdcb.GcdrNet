using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform;
using Avalonia.Platform.Storage;
using Sdcb.SimdGcdrNet;
using Sdcb.SimdGcdrNet.Models;
using AvaloniaBitmap = Avalonia.Media.Imaging.Bitmap;

namespace SkiaSharp.Avalonia;

public partial class MainWindow : Window
{
    private static readonly GcdrBackend[] s_backends = OperatingSystem.IsMacOS()
        ? [GcdrBackend.Auto, GcdrBackend.Cpu, GcdrBackend.Metal]
        : [GcdrBackend.Auto, GcdrBackend.Cpu, GcdrBackend.Vulkan];

    private readonly Dictionary<GcdrBackend, GcdrEnhancer> _enhancers = [];
    private SKBitmap? _source, _result;
    private string _imagePath;

    public MainWindow() : this(Program.StartupImagePath)
    {
    }

    public MainWindow(string imagePath)
    {
        InitializeComponent();
        _imagePath = imagePath;
        BackendCombo.ItemsSource = s_backends.Select(b => b.ToString().ToLowerInvariant()).ToArray();
        BackendCombo.SelectedIndex = 0;
        if (Design.IsDesignMode)
            return;

        Opened += async (_, _) =>
        {
            LoadImage(_imagePath);
            await EnhanceAsync();
        };
        Closed += (_, _) => DisposeResources();
    }

    private GcdrBackend SelectedBackend => s_backends[Math.Max(BackendCombo.SelectedIndex, 0)];

    private async void SelectImage_Click(object? sender, RoutedEventArgs e)
    {
        IReadOnlyList<IStorageFile> files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "选择文档照片",
            FileTypeFilter = [new FilePickerFileType("图片") { Patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"] }],
        });
        string? path = files.Count > 0 ? files[0].TryGetLocalPath() : null;
        if (path is null)
            return;

        LoadImage(path);
        await EnhanceAsync();
    }

    private async void Enhance_Click(object? sender, RoutedEventArgs e) => await EnhanceAsync();

    private async void Backend_SelectionChanged(object? sender, SelectionChangedEventArgs e)
    {
        if (IsVisible && _source is not null)
            await EnhanceAsync();
    }

    private async void Save_Click(object? sender, RoutedEventArgs e)
    {
        if (_result is null)
            return;

        IStorageFile? file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "保存增强结果",
            SuggestedFileName = Path.GetFileNameWithoutExtension(_imagePath) + "-enhanced.jpg",
            FileTypeChoices =
            [
                new FilePickerFileType("JPEG") { Patterns = ["*.jpg"] },
                new FilePickerFileType("PNG") { Patterns = ["*.png"] },
            ],
        });
        if (file is null)
            return;

        SKEncodedImageFormat format = file.Name.EndsWith(".png", StringComparison.OrdinalIgnoreCase)
            ? SKEncodedImageFormat.Png
            : SKEncodedImageFormat.Jpeg;
        using SKData data = _result.Encode(format, 92);
        await using Stream stream = await file.OpenWriteAsync();
        data.SaveTo(stream);
        Status.Text = $"已保存：{file.Name}";
    }

    private void LoadImage(string path)
    {
        try
        {
            SKBitmap bitmap = DecodeBgra(path);
            _source?.Dispose();
            _source = bitmap;
            _result?.Dispose();
            _result = null;
            _imagePath = path;
            ImagePathText.Text = path;
            SetPreview(SourcePreview, bitmap);
            SetPreview(ResultPreview, null);
            ResultPlaceholder.IsVisible = true;
            SaveButton.IsEnabled = false;
            Status.Text = $"{bitmap.Width}x{bitmap.Height}    {path}";
        }
        catch (Exception ex)
        {
            Status.Text = $"图片加载失败：{ex.Message}";
        }
    }

    private async Task EnhanceAsync()
    {
        SKBitmap? source = _source;
        if (source is null)
            return;

        GcdrBackend backend = SelectedBackend;
        SetBusy(true);
        Status.Text = $"正在增强（{backend.ToString().ToLowerInvariant()}）…";
        try
        {
            (SKBitmap result, double ms) = await Task.Run(() =>
            {
                GcdrEnhancer enhancer = GetEnhancer(backend);
                SKBitmap dst = new(source.Info);
                Stopwatch sw = Stopwatch.StartNew();
                unsafe
                {
                    // SKBitmap pixels are BGRA32 rows of RowBytes; Enhance reads/writes them directly.
                    enhancer.Enhance(
                        new ReadOnlySpan<byte>((byte*)source.GetPixels(), source.RowBytes * source.Height),
                        source.Width, source.Height, source.RowBytes,
                        new Span<byte>((byte*)dst.GetPixels(), dst.RowBytes * dst.Height),
                        dst.RowBytes, GcdrPixelFormat.Bgra32);
                }
                return (dst, sw.Elapsed.TotalMilliseconds);
            });

            if (source != _source)
            {
                result.Dispose();
                return;
            }
            _result?.Dispose();
            _result = result;
            SetPreview(ResultPreview, result);
            ResultPlaceholder.IsVisible = false;
            SaveButton.IsEnabled = true;
            Status.Text = $"{source.Width}x{source.Height}    {backend.ToString().ToLowerInvariant()}    增强 {ms:F1} ms" +
                "    （每个新尺寸第一次包含计划编译，再点一次「增强」可看到稳态耗时）";
        }
        catch (Exception ex)
        {
            Status.Text = $"增强失败（{backend.ToString().ToLowerInvariant()}）：{ex.Message}";
        }
        finally
        {
            SetBusy(false);
        }
    }

    private GcdrEnhancer GetEnhancer(GcdrBackend backend)
    {
        lock (_enhancers)
        {
            if (!_enhancers.TryGetValue(backend, out GcdrEnhancer? enhancer))
                _enhancers[backend] = enhancer = GcdrEnhancer.FromBytes(GcdrModels.GcNet, GcdrModels.DrNet, backend);
            return enhancer;
        }
    }

    private void SetBusy(bool busy)
    {
        RunButton.IsEnabled = !busy;
        SelectImageButton.IsEnabled = !busy;
        BackendCombo.IsEnabled = !busy;
    }

    private static void SetPreview(Image image, SKBitmap? bitmap)
    {
        (image.Source as IDisposable)?.Dispose();
        image.Source = bitmap is null
            ? null
            : new AvaloniaBitmap(PixelFormat.Bgra8888, AlphaFormat.Unpremul, bitmap.GetPixels(),
                new PixelSize(bitmap.Width, bitmap.Height), new Vector(96, 96), bitmap.RowBytes);
    }

    private static SKBitmap DecodeBgra(string path)
    {
        SKBitmap bitmap = SKBitmap.Decode(path) ?? throw new InvalidDataException($"无法读取图片：{path}");
        if (bitmap.ColorType == SKColorType.Bgra8888)
            return bitmap;

        SKBitmap? converted = bitmap.Copy(SKColorType.Bgra8888);
        bitmap.Dispose();
        return converted ?? throw new InvalidDataException("无法转换到 BGRA。");
    }

    private void DisposeResources()
    {
        SetPreview(SourcePreview, null);
        SetPreview(ResultPreview, null);
        _source?.Dispose();
        _result?.Dispose();
        foreach (GcdrEnhancer enhancer in _enhancers.Values)
            enhancer.Dispose();
        _enhancers.Clear();
    }
}
