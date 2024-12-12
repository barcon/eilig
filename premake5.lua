-- premake5.lua
workspace "eilig"
   configurations { "Debug", "Release", "Python" }
   location "build"

project "eilig"
   kind "StaticLib"
   language "C++"
   cppdialect "C++17"
   
   targetdir "build/%{cfg.buildcfg}"
   includedirs { "../utils/src"}
   includedirs { "../logger/src"}
   includedirs { "../opencl/inc"}

   files { "src/**.hpp", "src/**.cpp" }

   filter "configurations:Debug"
	  architecture "x86_64"    
	  defines { "DEBUG" }
      symbols "On"
	  removefiles "eilig_export_python.cpp"

   filter "configurations:Release"
      architecture "x86_64" 	  
	  defines { "NDEBUG" }
      optimize "Speed"
	  removefiles "eilig_export_python.cpp"	  
	  
   filter "configurations:Python"
      architecture "x86_64"  
	  defines { "NDEBUG" }
      optimize "Speed"	  