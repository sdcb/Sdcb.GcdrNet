using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using OpenCvSharp;
using Sdcb.SimdGcdrNet;
using Sdcb.SimdGcdrNet.Models;

namespace OpenCvSharp5.Wpf;

public partial class MainWindow : System.Windows.Window
{
    private static readonly GcdrBackend[] s_backends = [GcdrBackend.Auto, GcdrBackend.Cpu, GcdrBackend.Vulkan];

    private readonly Dictionary<GcdrBackend, GcdrEnhancer> _enhancers = [];
    private Mat? _source, _result;
    private string _imagePath;

    public MainWindow(string imagePath)
    {
        InitializeComponent();
        _imagePath = imagePath;
        BackendCombo.ItemsSource = s_backends.Select(b => b.ToString().ToLowerInvariant()).ToArray();
        BackendCombo.SelectedIndex = 0;
        Loaded += async (_, _) =>
        {
            LoadImage(_imagePath);
            await EnhanceAsync();
        };
        Closed += (_, _) => DisposeResources();
    }

    private GcdrBackend SelectedBackend => s_backends[Math.Max(BackendCombo.SelectedIndex, 0)];

    private async void SelectImage_Click(object sender, RoutedEventArgs e)
    {
        OpenFileDialog dialog = new()
        {
            Title = "选择文档照片",
            Filter = "图片文件|*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tif;*.tiff|所有文件|*.*",
        };
        if (dialog.ShowDialog(this) != true)
            return;

        LoadImage(dialog.FileName);
        await EnhanceAsync();
    }

    private async void Window_Drop(object sender, DragEventArgs e)
    {
        if (e.Data.GetData(DataFormats.FileDrop) is string[] { Length: > 0 } files)
        {
            LoadImage(files[0]);
            await EnhanceAsync();
        }
    }

    private async void Enhance_Click(object sender, RoutedEventArgs e) => await EnhanceAsync();

    private async void Backend_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (IsLoaded && _source is not null)
            await EnhanceAsync();
    }

    private void Save_Click(object sender, RoutedEventArgs e)
    {
        if (_result is null)
            return;

        SaveFileDialog dialog = new()
        {
            Title = "保存增强结果",
            FileName = Path.GetFileNameWithoutExtension(_imagePath) + "-enhanced.jpg",
            Filter = "JPEG|*.jpg|PNG|*.png",
        };
        if (dialog.ShowDialog(this) != true)
            return;

        // ImEncode + File.WriteAllBytes instead of ImWrite: OpenCV's own file I/O cannot open non-ASCII paths on Windows.
        Cv2.ImEncode(Path.GetExtension(dialog.FileName), _result, out byte[] encoded);
        File.WriteAllBytes(dialog.FileName, encoded);
        Status.Text = $"已保存：{dialog.FileName}";
    }

    private void LoadImage(string path)
    {
        try
        {
            Mat image = Cv2.ImDecode(File.ReadAllBytes(path), ImreadModes.Color);
            if (image.Empty())
            {
                image.Dispose();
                throw new InvalidDataException($"无法读取图片：{path}");
            }

            _source?.Dispose();
            _source = image;
            _result?.Dispose();
            _result = null;
            _imagePath = path;
            ImagePathText.Text = path;
            SourcePreview.Source = ToBitmapSource(image);
            ResultPreview.Source = null;
            ResultPlaceholder.Visibility = Visibility.Visible;
            SaveButton.IsEnabled = false;
            Status.Text = $"{image.Width}x{image.Height}    {path}";
        }
        catch (Exception ex)
        {
            Status.Text = $"图片加载失败：{ex.Message}";
        }
    }

    private async Task EnhanceAsync()
    {
        Mat? source = _source;
        if (source is null)
            return;

        GcdrBackend backend = SelectedBackend;
        string backendName = backend.ToString().ToLowerInvariant();
        SetBusy(true);
        Status.Text = $"正在增强（{backendName}）…";
        try
        {
            (Mat result, double ms) = await Task.Run(() =>
            {
                GcdrEnhancer enhancer = GetEnhancer(backend);
                Mat dst = new(source.Rows, source.Cols, MatType.CV_8UC3);
                Stopwatch sw = Stopwatch.StartNew();
                unsafe
                {
                    // Mat (CV_8UC3) is BGR24 rows of Step() bytes; Enhance reads/writes them directly.
                    int srcStride = checked((int)source.Step()), dstStride = checked((int)dst.Step());
                    enhancer.Enhance(
                        new ReadOnlySpan<byte>((byte*)source.Data, srcStride * source.Rows),
                        source.Cols, source.Rows, srcStride,
                        new Span<byte>((byte*)dst.Data, dstStride * dst.Rows),
                        dstStride, GcdrPixelFormat.Bgr24);
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
            ResultPreview.Source = ToBitmapSource(result);
            ResultPlaceholder.Visibility = Visibility.Collapsed;
            SaveButton.IsEnabled = true;
            Status.Text = $"{source.Width}x{source.Height}    {backendName}    增强 {ms:F1} ms" +
                "    （每个新尺寸第一次包含计划编译，再点一次「增强」可看到稳态耗时）";
        }
        catch (Exception ex)
        {
            Status.Text = $"增强失败（{backendName}）：{ex.Message}";
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
        AllowDrop = !busy;
    }

    private static BitmapSource ToBitmapSource(Mat image)
    {
        int stride = checked((int)image.Step());
        BitmapSource bitmap = BitmapSource.Create(image.Width, image.Height, 96, 96, PixelFormats.Bgr24, null,
            image.Data, stride * image.Height, stride);
        bitmap.Freeze();
        return bitmap;
    }

    private void DisposeResources()
    {
        _source?.Dispose();
        _result?.Dispose();
        foreach (GcdrEnhancer enhancer in _enhancers.Values)
            enhancer.Dispose();
        _enhancers.Clear();
    }
}
