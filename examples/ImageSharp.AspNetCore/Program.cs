using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using ImageSharp.AspNetCore;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Mvc;
using Scalar.AspNetCore;
using Sdcb.SimdGcdrNet;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Formats.Bmp;
using SixLabors.ImageSharp.Formats.Gif;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.Formats.Webp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

const long maxUploadBytes = 20 * 1024 * 1024;
string[] timingHeaders = ["X-Decode-Ms", "X-Enhance-Ms", "X-Encode-Ms", "X-Image-Size", "X-Backend"];

WebApplicationBuilder builder = WebApplication.CreateBuilder(args);
builder.WebHost.ConfigureKestrel(options => options.Limits.MaxRequestBodySize = maxUploadBytes);
builder.Services.Configure<FormOptions>(options => options.MultipartBodyLengthLimit = maxUploadBytes);
builder.Services.AddSingleton<GcdrEngine>();
builder.Services.AddOpenApi();
builder.Services.AddCors(options => options.AddDefaultPolicy(policy =>
    policy.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod().WithExposedHeaders(timingHeaders)));

WebApplication app = builder.Build();
app.UseCors();
app.UseDefaultFiles();
app.UseStaticFiles();
app.MapOpenApi();
app.MapScalarApiReference();

app.MapGet("/api/backends", () => Results.Ok(new BackendList(GcdrEngine.Backends, GcdrEngine.DefaultBackend)))
    .WithName("ListBackends")
    .WithTags("Enhance")
    .WithSummary("列出可用计算后端")
    .Produces<BackendList>();

app.MapGet("/example.jpg", (IWebHostEnvironment env) =>
{
    string? path = new[]
    {
        Path.Combine(AppContext.BaseDirectory, "example.jpg"),
        Path.GetFullPath(Path.Combine(env.ContentRootPath, "..", "example.jpg")),
    }.FirstOrDefault(File.Exists);
    return path is null ? Results.NotFound() : Results.File(path, "image/jpeg");
}).ExcludeFromDescription();

app.MapPost("/api/enhance", EnhanceAsync)
    .DisableAntiforgery()
    .WithName("EnhanceImage")
    .WithTags("Enhance")
    .WithSummary("文档图片去阴影 / 外观增强")
    .WithDescription("上传图片（表单字段 file），可选 backend=auto|cpu、format=jpeg|webp|png（默认 jpeg）。" +
        "返回增强后的图片；耗时与尺寸见响应头 X-Decode-Ms、X-Enhance-Ms、X-Encode-Ms、X-Image-Size、X-Backend。" +
        "每个新尺寸的第一次调用包含计划编译，耗时偏高。")
    .Produces(StatusCodes.Status200OK, contentType: "image/jpeg")
    .Produces<EnhanceError>(StatusCodes.Status400BadRequest)
    .Produces<EnhanceError>(StatusCodes.Status500InternalServerError)
    .Accepts<EnhanceUpload>("multipart/form-data");

app.Run();

static async Task<IResult> EnhanceAsync(
    [FromForm] EnhanceUpload upload,
    GcdrEngine engine,
    HttpResponse response,
    CancellationToken cancellationToken)
{
    IFormFile? file = upload.File;
    if (file is null || file.Length == 0)
        return Results.BadRequest(new EnhanceError("请上传图片文件（表单字段 file）。"));
    if (file.Length > maxUploadBytes)
        return Results.BadRequest(new EnhanceError("图片超过 20 MB 上限。"));
    if (!GcdrEngine.TryParseBackend(upload.Backend, out GcdrBackend backend))
        return Results.BadRequest(new EnhanceError("backend 必须是 auto 或 cpu。"));
    if (!TryGetEncoder(upload.Format,     out IImageEncoder encoder, out string contentType))
        return Results.BadRequest(new EnhanceError("format 必须是 jpeg、webp 或 png。"));

    Stopwatch stage = Stopwatch.StartNew();
    try
    {
        await using Stream stream = file.OpenReadStream();
        using Image<Rgba32> image = await Image.LoadAsync<Rgba32>(GcdrImageDecode.Options, stream, cancellationToken);
        if ((long)image.Width * image.Height > GcdrEngine.MaxImagePixels)
            return Results.BadRequest(new EnhanceError("图片像素超过上限（1600 万）。"));
        // Phone photos usually carry an EXIF rotation; bake it in so the output looks like the input.
        image.Mutate(x => x.AutoOrient());
        double decodeMs = stage.Elapsed.TotalMilliseconds;

        stage.Restart();
        GcdrEnhancer enhancer = engine.Get(backend);
        int stride = image.Width * 4;
        if (image.DangerousTryGetSinglePixelMemory(out Memory<Rgba32> memory))
        {
            // In place: src and dst are the same RGBA buffer with the same stride.
            Span<byte> pixels = MemoryMarshal.AsBytes(memory.Span);
            enhancer.Enhance(pixels, image.Width, image.Height, stride, pixels, stride, GcdrPixelFormat.Rgba32);
        }
        else
        {
            Rgba32[] packed = new Rgba32[checked(image.Width * image.Height)];
            image.CopyPixelDataTo(packed);
            Span<byte> pixels = MemoryMarshal.AsBytes(packed.AsSpan());
            enhancer.Enhance(pixels, image.Width, image.Height, stride, pixels, stride, GcdrPixelFormat.Rgba32);
            image.Frames.RootFrame.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                    packed.AsSpan(y * accessor.Width, accessor.Width).CopyTo(accessor.GetRowSpan(y));
            });
        }
        double enhanceMs = stage.Elapsed.TotalMilliseconds;

        stage.Restart();
        MemoryStream output = new();
        await image.SaveAsync(output, encoder, cancellationToken);
        double encodeMs = stage.Elapsed.TotalMilliseconds;
        output.Position = 0;

        response.Headers["X-Decode-Ms"] = FormatMs(decodeMs);
        response.Headers["X-Enhance-Ms"] = FormatMs(enhanceMs);
        response.Headers["X-Encode-Ms"] = FormatMs(encodeMs);
        response.Headers["X-Image-Size"] = $"{image.Width}x{image.Height}";
        response.Headers["X-Backend"] = backend.ToString().ToLowerInvariant();
        return Results.Stream(output, contentType);
    }
    catch (UnknownImageFormatException)
    {
        return Results.BadRequest(new EnhanceError("无法解析图片，请上传 JPEG、PNG、WebP、BMP 或 GIF。"));
    }
    catch (InvalidImageContentException ex)
    {
        return Results.BadRequest(new EnhanceError($"图片内容无效：{ex.Message}"));
    }
    catch (Exception ex)
    {
        return Results.Json(new EnhanceError(ex.Message), statusCode: StatusCodes.Status500InternalServerError);
    }
}

static bool TryGetEncoder(string? format, out IImageEncoder encoder, out string contentType)
{
    (encoder, contentType) = (string.IsNullOrWhiteSpace(format) ? "jpeg" : format.Trim().ToLowerInvariant()) switch
    {
        "webp" => ((IImageEncoder)new WebpEncoder { FileFormat = WebpFileFormatType.Lossy, Quality = 90 }, "image/webp"),
        "jpeg" or "jpg" => (new JpegEncoder { Quality = 90 }, "image/jpeg"),
        "png" => (new PngEncoder(), "image/png"),
        _ => (null!, ""),
    };
    return encoder is not null;
}

static string FormatMs(double ms) => ms.ToString("0.0", CultureInfo.InvariantCulture);

internal sealed class EnhanceUpload
{
    public IFormFile? File { get; init; }

    [Description("auto（GPU 优先：Vulkan / Metal，无 GPU 时回退 CPU）或 cpu，默认 auto。")]
    public string? Backend { get; init; }

    [Description("输出格式：jpeg（默认）、webp 或 png。")]
    public string? Format { get; init; }
}

internal sealed record BackendList(string[] Backends, string Default);

internal sealed record EnhanceError(string Error);

// Only the formats this demo needs: no TIFF decoder, which is where most ImageSharp 3.1.x advisories live.
// A new Configuration (not Configuration.Default) so the global defaults stay untouched.
file static class GcdrImageDecode
{
    public static readonly DecoderOptions Options = CreateOptions();

    static DecoderOptions CreateOptions()
    {
        Configuration configuration = new(
            new JpegConfigurationModule(),
            new PngConfigurationModule(),
            new WebpConfigurationModule(),
            new BmpConfigurationModule(),
            new GifConfigurationModule())
        {
            // The default allocator splits pixels into 4 MB chunks; ask for one buffer so Enhance can run in place.
            PreferContiguousImageBuffers = true,
        };
        return new DecoderOptions { Configuration = configuration };
    }
}
