using LLama;
using LLama.Common;
using LLama.Sampling;
using LLama.Transformers;

Console.Write("guff file path:");
var modelPath = Console.ReadLine().Trim('\"');
var parameters = new ModelParams(modelPath)
{
    GpuLayerCount = 15
};
using var model = LLamaWeights.LoadFromFile(parameters);
using var context = model.CreateContext(parameters);
var ex = new InteractiveExecutor(context); 

ChatHistory chatHistory = new ChatHistory();
chatHistory.AddMessage(AuthorRole.System, "あなたは優秀なアシスタントです。どんなことでも的確に答える必要があり、間違えることは許されません。しかしながら、間違えてしまった場合には即座に認め訂正してください");
chatHistory.AddMessage(AuthorRole.User, "こんにちは");
chatHistory.AddMessage(AuthorRole.Assistant, "お手伝いが必要ですか？");
var inferenceParams = new InferenceParams
{
    SamplingPipeline = new DefaultSamplingPipeline
    {
        Temperature = 0.9f
    },
    AntiPrompts = new List<string> { "User:" },
    MaxTokens=-1
};

var chatSession = new ChatSession(ex, chatHistory);
chatSession.WithHistoryTransform(new PromptTemplateTransformer(model, withAssistant: true));
chatSession.WithOutputTransform(new LLamaTransforms.KeywordTextOutputStreamTransform(
    ["User:", "�"],
    redundancyLength: 5));
while (true)
{
    Console.Write("User>");
    string prompt = Console.ReadLine();
    Console.Write("Assistant>");

    await foreach (var text in chatSession.ChatAsync(
        new ChatHistory.Message(AuthorRole.User, prompt), inferenceParams))
    {
        Console.ForegroundColor = ConsoleColor.White;
        Console.Write(text);
    }
}