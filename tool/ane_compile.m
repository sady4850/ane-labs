// ane-compile DIR [--load]: compiles DIR/model.mil (with DIR/weights.bin) through the private
// AppleNeuralEngine service, optionally loads and unloads it, purges the compilation and prints
// one JSON line: {"ok": bool, "stage": ..., "compile_ms": ..., "load_ms": ..., "error": ...}.

#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <objc/runtime.h>
#include <sys/qos.h>
#include <time.h>

@protocol AneModel
+ (id)modelAtURL:(NSURL *)url key:(NSString *)key;
@end
@protocol AneClient
+ (id)sharedConnection;
- (BOOL)compileModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)compiledModelExistsFor:(id)model;
- (void)purgeCompiledModel:(id)model;
- (BOOL)loadModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)unloadModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
@end

static double now_ms(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6; }

static void emit(BOOL ok, const char *stage, double compile, double load, NSString *error) {
  NSMutableDictionary *out = [@{@"ok" : @(ok), @"stage" : @(stage)} mutableCopy];
  if (compile >= 0) out[@"compile_ms"] = @((long)llround(compile));
  if (load >= 0) out[@"load_ms"] = @((long)llround(load));
  if (error) out[@"error"] = error;
  NSData *json = [NSJSONSerialization dataWithJSONObject:out options:NSJSONWritingSortedKeys error:nil];
  printf("%s\n", [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
}

static NSString *describe(NSError *error) {
  if (!error) return @"no error";
  NSMutableString *text = [NSMutableString stringWithString:error.localizedDescription ?: @""];
  for (NSError *under = error.userInfo[NSUnderlyingErrorKey]; under; under = under.userInfo[NSUnderlyingErrorKey])
    [text appendFormat:@" | %@", under.localizedDescription];
  return text;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2) {
      fprintf(stderr, "usage: ane-compile DIR [--load]\n");
      return 2;
    }
    const BOOL load = argc > 2 && !strcmp(argv[2], "--load");
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
      emit(NO, "framework", -1, -1, @(dlerror() ?: "dlopen failed"));
      return 1;
    }
    Class modelClass = NSClassFromString(@"_ANEModel"), clientClass = NSClassFromString(@"_ANEClient");
    if (!modelClass || !clientClass) {
      emit(NO, "framework", -1, -1, @"no _ANEModel or _ANEClient");
      return 1;
    }
    NSString *dir = [@(argv[1]) stringByStandardizingPath];
    id<AneClient> client = [(Class<AneClient>)clientClass sharedConnection];
    id model = [(Class<AneModel>)modelClass modelAtURL:[NSURL fileURLWithPath:dir isDirectory:YES]
                                                   key:dir.lastPathComponent];
    if (!client || !model) {
      emit(NO, "model", -1, -1, @"no client or model");
      return 1;
    }
    const unsigned qos = QOS_CLASS_DEFAULT;
    if ([client compiledModelExistsFor:model]) [client purgeCompiledModel:model];
    NSError *error = nil;
    double start = now_ms();
    BOOL ok = NO;
    @try {
      ok = [client compileModel:model
                        options:@{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"}
                            qos:qos
                          error:&error];
    } @catch (NSException *exception) {
      emit(NO, "compile", now_ms() - start, -1, [NSString stringWithFormat:@"%@: %@", exception.name, exception.reason]);
      return 1;
    }
    const double compile = now_ms() - start;
    if (!ok) {
      emit(NO, "compile", compile, -1, describe(error));
      [client purgeCompiledModel:model];
      return 1;
    }
    double loaded = -1;
    if (load) {
      start = now_ms();
      ok = [client loadModel:model options:@{} qos:qos error:&error];
      loaded = now_ms() - start;
      if (!ok) {
        emit(NO, "load", compile, loaded, describe(error));
        [client purgeCompiledModel:model];
        return 1;
      }
      [client unloadModel:model options:@{} qos:qos error:nil];
    }
    [client purgeCompiledModel:model];
    emit(YES, load ? "load" : "compile", compile, loaded, nil);
    return 0;
  }
}
