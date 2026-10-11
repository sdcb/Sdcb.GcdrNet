<Query Kind="Program">
  <Namespace>System.Threading.Tasks</Namespace>
  <Namespace>LINQPad.Controls</Namespace>
  <IncludeUncapsulator>false</IncludeUncapsulator>
</Query>

#load ".\00-common"

DumpContainer dc = new DumpContainer().Dump();

async Task Main()
{
	await SetupAsync(QueryCancelToken);
	Refresh();
}

void Refresh()
{
	dc.Content = LoadTable();
}

object LoadTable()
{
	return new
	{
		Functions = Util.HorizontalRun(true,
			new Button("Build All", _ => Projects.ToList().ForEach(Build)),
			new Button("Clear Cache", _ => ClearNuGetCache()),
			new Button("📂Open nupkgs", _ => Process.Start("explorer", Path.GetFullPath(@".\nupkgs")))),
		Table = Projects
			.Select(x => new
			{
				Project = x.name,
				Version = x.version,
				Build = new Button("Build", o => Build(x))
			})
	};
}

// Makes a re-published package (even with the same version) visible to the next restore:
// clears NuGet's feed-metadata cache and removes only this repo's packages from the
// global-packages folder; every other package stays.
void ClearNuGetCache()
{
	DotNetRun("nuget locals http-cache --clear");
	DotNetRun("nuget locals temp --clear");
	string root = GlobalPackagesFolder();
	foreach (ProjectVersion p in Projects)
	{
		string dir = Path.Combine(root, p.name.ToLowerInvariant());
		if (!Directory.Exists(dir)) continue;
		Directory.Delete(dir, recursive: true);
		$"deleted {dir}".Dump();
	}
}

// honours NUGET_PACKAGES / nuget.config overrides
static string GlobalPackagesFolder()
{
	using Process p = Process.Start(new ProcessStartInfo("dotnet", "nuget locals global-packages --list")
	{
		RedirectStandardOutput = true,
		UseShellExecute = false,
	})!;
	string output = p.StandardOutput.ReadToEnd();
	p.WaitForExit();
	const string prefix = "global-packages:";
	string? line = output.Split('\n').Select(x => x.Trim()).FirstOrDefault(x => x.StartsWith(prefix, StringComparison.OrdinalIgnoreCase));
	if (line is null) throw new Exception($"Cannot locate the global-packages folder:{Environment.NewLine}{output}");
	return line[prefix.Length..].Trim();
}

void Build(ProjectVersion p)
{
	QueryCancelToken.ThrowIfCancellationRequested();
	string projPosition = FindProjectPath(p.name);
	// No -p:Version= (see 00-common); each project's version comes from AllVersionMsBuildArgs().
	DotNetRun($@"pack ""{projPosition}"" -c Release -o .\nupkgs {AllVersionMsBuildArgs()}");
	Refresh();
}
